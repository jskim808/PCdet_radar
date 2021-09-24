import copy
import pickle
from pathlib import Path
import os.path as osp
import os
import shutil

import numpy as np
from tqdm import tqdm

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate
from ...core.box_np_ops import points_in_rbbox
from ...core.box_np_ops import points_count_rbbox


class NuScenesRADARDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.radar_version = dataset_cfg.RADAR_VERSION
        self.filter_version = dataset_cfg.FILTER_VERSION
        self.infos = []
        self.include_nuscenes_data(self.mode)
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)

    def include_nuscenes_data(self, mode):
        self.logger.info('Loading NuScenes dataset')
        nuscenes_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                nuscenes_infos.extend(infos)

        self.infos.extend(nuscenes_infos)
        self.logger.info('Total samples for NuScenes dataset: %d' % (len(nuscenes_infos)))

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos_new[name].append(info)

        cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}

        return sampled_infos
    
    def get_radar_data(self, index):
        info = self.infos[index]
        radar_path = info['radar_path']
        points = np.load(radar_path)
        return points

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_radar_data(index)

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'radar_frame_id' : Path(info['radar_path']).stem,
            'metadata': {'token': info['token']}
        }

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_radar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })
        # input_gt_num = len(input_dict['gt_names'])

        # input_dict['gt_boxes'].shape = [N,7]
        data_dict = self.prepare_data(data_dict=input_dict)
        # data_dict['gt_boxes'].shape = [N,8] --> data_dict['gt_boxes'][:,-1] is gt class!!
        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False):
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]
        
        # data_gt_num = data_dict['gt_boxes'].shape[1]
        # print(input_gt_num == data_gt_num)
        # import pdb; pdb.set_trace()
        return data_dict


    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        import json
        from nuscenes.nuscenes import NuScenes
        from . import nuscenes_utils
        nusc = NuScenes(version='v1.0-trainval', dataroot='/mnt/sda/jskim/OpenPCDet/data/nuscenes/v1.0-trainval', verbose=True)
        eval_x20 = self.dataset_cfg.EVALUATION_X20
        if eval_x20:
            eval_range = '20'
        else:
            eval_range = '50'
        
        nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc, eval_x20)

        gt_bboxes,gt_lidar = nuscenes_utils.ground_truth_bbox(self.infos,self.class_names,eval_x20 ,nusc)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': False,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc.json')
        second_gt_path = '/home/spalab/jskim/Jisong_RADAR_base_july/second/experiment/experiment_9/vel_rel_base/eval_results/step_4219/gt_nusc.json'
        gt_path = Path(output_path) / f"gt_nusc.json"
        gt_path_2 = Path(output_path) / f"gt_lidar_nusc.json"
        if osp.exists(f'{gt_path}'):
            os.remove(f'{gt_path}')
        # if not osp.exists(f'{gt_path}'):
        #     with open(gt_path, "w") as f2:
        #         json.dump(gt_bboxes, f2)
        # import pdb; pdb.set_trace()
        shutil.copyfile(second_gt_path,gt_path)
        if not osp.exists(f'{gt_lidar}'):
            with open(gt_path_2, "w") as f3:
                json.dump(gt_lidar, f3)
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)

        self.logger.info(f'The predictions of NuScenes have been saved to {res_path}')

        if self.dataset_cfg.VERSION == 'v1.0-test':
            return 'No ground-truth annotations for evaluation', {}

        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v1.0-test': 'test'
        }
        try:
            eval_version = 'detection_cvpr_2019'
            eval_config = config_factory(eval_version)
        except:
            eval_version = 'cvpr_2019'
            eval_config = config_factory(eval_version)

        nusc_eval = NuScenesEval(
            nusc,
            config=eval_config,
            result_path=res_path,
            eval_set=eval_set_map[self.dataset_cfg.VERSION],
            output_dir=str(output_path),
            verbose=True,
        )
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

        with open(output_path / 'metrics_summary.json', 'r') as f:
            metrics = json.load(f)

        result_str, result_dict = nuscenes_utils.format_nuscene_results(metrics, self.class_names, version=eval_version)
        return result_str, result_dict

    def create_groundtruth_database(self, used_classes=None, max_sweeps=10):
        import torch

        database_save_path = self.root_path / f'gt_database_{self.filter_version}_{self.radar_version}_sweeps{max_sweeps}'
        db_info_save_path = self.root_path / f"dbinfos_{self.filter_version}_{self.radar_version}_sweeps{max_sweeps}.pkl"

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        for idx in tqdm(range(len(self.infos))):
            sample_idx = idx
            info = self.infos[idx]
            points = self.get_radar_data(idx)
            gt_boxes = info['gt_boxes']
            gt_second_boxes = info['gt_second_boxes']
            gt_names = info['gt_names']

            ref_boxes = copy.deepcopy(gt_boxes)
            ref_boxes[:,2]=0
            ref_boxes[:,5]=10

            indixes_2 = points_in_rbbox(points,gt_second_boxes)
            point_idxes, box_idxes = np.where(indixes_2 != False)
            assert(len(point_idxes),len(box_idxes))
            # import pdb; pdb.set_trace()
            pts_ = np.ones(len(points),dtype=np.int8)*-1
            for i,ptx in enumerate(point_idxes):
                box_idx = box_idxes[i]
                pts_[ptx] = box_idx
            box_idxs_of_pts_2 = pts_
            # box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
            #     torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
            #     torch.from_numpy(ref_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            # ).long().squeeze(dim=0).cpu().numpy()

            for i in range(gt_boxes.shape[0]):
                filename = '%s_%s_%d.npy' % (sample_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts_2 == i]
                gt_points[:, :3] -= gt_boxes[i, :3]
                np.save(filepath, gt_points)

                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.npy
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


class NuScenesRadarDatasetD2(NuScenesRADARDataset):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if len(self.infos) > 28000:
            self.infos = list(
                sorted(self.infos, key=lambda e: e["timestamp"]))
            self.infos = self.infos[::2]

def create_nuscenes_info(version, data_path, save_path, filter_version, radar_version, max_sweeps=10):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from nuscenes.utils import splits
    from . import nuscenes_utils
    data_path = data_path / version
    save_path = save_path / version

    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise NotImplementedError

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    nusc_can = NuScenesCanBus(dataroot=data_path)
    available_scenes = nuscenes_utils._get_available_scenes(nusc,nusc_can)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    future_tokens = []
    for train_scene_tk in train_scenes:
        trainscene = nusc.get('scene',train_scene_tk)
        first_sample = nusc.get('sample',trainscene['first_sample_token'])
        for _ in range(2):
            future_tokens.append(first_sample['token'])
            first_sample = nusc.get('sample',first_sample['next'])
    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = nuscenes_utils._fill_trainval_infos(
        data_path=data_path, nusc=nusc, nusc_can= nusc_can, train_scenes=train_scenes, val_scenes=val_scenes,
        radar_version=radar_version, future_tokens=future_tokens, filter_version=filter_version,
        test='test' in version, max_sweeps=max_sweeps
    )

    if version == 'v1.0-test':
        print('test sample: %d' % len(train_nusc_infos))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_test.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print('train sample: %d, val sample: %d' % (len(train_nusc_infos), len(val_nusc_infos)))
        with open(save_path / f'infos_train_{filter_version}_{radar_version}_sweeps{max_sweeps}.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(save_path / f'infos_val_{filter_version}_{radar_version}_sweeps{max_sweeps}.pkl', 'wb') as f:
            pickle.dump(val_nusc_infos, f)


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_nuscenes_infos', help='')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    args = parser.parse_args()

    if args.func == 'create_nuscenes_infos':
        dataset_cfg = EasyDict(yaml.load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        DATA_DIR = (Path(__file__).resolve().parent / '/mnt/sda/jskim').resolve()
        dataset_cfg.VERSION = args.version
        
        create_nuscenes_info(
            version=dataset_cfg.VERSION,
            data_path=DATA_DIR /'OpenPCDet'/'data'/'nuscenes',
            save_path=ROOT_DIR / 'data' / 'nuscenes_radar_0831',
            filter_version = dataset_cfg.FILTER_VERSION,
            radar_version = dataset_cfg.RADAR_VERSION,
            max_sweeps=dataset_cfg.MAX_SWEEPS,
        )
        
        nuscenes_dataset = NuScenesRADARDataset(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=ROOT_DIR / 'data' / 'nuscenes_radar_0831',
            logger=common_utils.create_logger(), training=True
        )
        nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)
