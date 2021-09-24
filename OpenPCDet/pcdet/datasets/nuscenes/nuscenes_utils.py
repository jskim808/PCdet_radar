"""
The NuScenes data pre-processing and evaluation is modified from
https://github.com/traveller59/second.pytorch and https://github.com/poodarchu/Det3D
"""

import operator
from functools import reduce
from pathlib import Path
import os
from copy import deepcopy
import copy
import os.path as osp
import pickle

import numpy as np
import tqdm
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from ...utils import box_utils
from ...core.box_np_ops import points_in_rbbox
from ...core.box_np_ops import points_count_rbbox

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    # 'human.pedestrian.wheelchair': 'ignore',
    # 'human.pedestrian.stroller': 'ignore',
    # 'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    # 'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    # 'vehicle.emergency.ambulance': 'ignore',
    # 'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    # 'movable_object.pushable_pullable': 'ignore',
    # 'movable_object.debris': 'ignore',
    # 'static_object.bicycle_rack': 'ignore',
}


cls_attr_dist = {
    'barrier': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'bicycle': {
        'cycle.with_rider': 2791,
        'cycle.without_rider': 8946,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'bus': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 9092,
        'vehicle.parked': 3294,
        'vehicle.stopped': 3881,
    },
    'car': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 114304,
        'vehicle.parked': 330133,
        'vehicle.stopped': 46898,
    },
    'construction_vehicle': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 882,
        'vehicle.parked': 11549,
        'vehicle.stopped': 2102,
    },
    'ignore': {
        'cycle.with_rider': 307,
        'cycle.without_rider': 73,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 165,
        'vehicle.parked': 400,
        'vehicle.stopped': 102,
    },
    'motorcycle': {
        'cycle.with_rider': 4233,
        'cycle.without_rider': 8326,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'pedestrian': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 157444,
        'pedestrian.sitting_lying_down': 13939,
        'pedestrian.standing': 46530,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'traffic_cone': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'trailer': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 3421,
        'vehicle.parked': 19224,
        'vehicle.stopped': 1895,
    },
    'truck': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 21339,
        'vehicle.parked': 55626,
        'vehicle.stopped': 11097,
    },
}


def _get_available_samples(nusc):
    available_samples = []
    can_black_list = [161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 309, 310, 311, 312, 313, 314]
    for i in range(len(nusc.sample)):
        scene_tk = nusc.sample[i]['scene_token']
        scene_mask = int(nusc.get('scene',scene_tk)['name'][-4:])
        if scene_mask in can_black_list:
            continue
        else:
            available_samples.append(nusc.sample[i])
    return available_samples

def _get_available_scenes(nusc, nusc_can):
    available_scenes = []
    can_black_list = [161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 309, 310, 311, 312, 313, 314]
    print("total scene num:", len(nusc.scene))
    for scene in nusc.scene:
        scene_id = int(scene['name'][-4:])
        if scene_id in can_black_list:
            continue
        else:
            scene_token = scene["token"]
            scene_rec = nusc.get('scene', scene_token)
            sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
            sd_rec = nusc.get('sample_data', sample_rec['data']["LIDAR_TOP"])
            has_more_frames = True
            scene_not_exist = False
            while has_more_frames:
                lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
                if not Path(lidar_path).exists():
                    scene_not_exist = True
                    break
                else:
                    break
                if not sd_rec['next'] == "":
                    sd_rec = nusc.get('sample_data', sd_rec['next'])
                else:
                    has_more_frames = False
            if scene_not_exist:
                continue
            available_scenes.append(scene)
    print("exist scene num:", len(available_scenes))
    return available_scenes

def get_available_scenes(nusc):
    available_scenes = []
    print('total scene num:', len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            # if not sd_rec['next'] == '':
            #     sd_rec = nusc.get('sample_data', sd_rec['next'])
            # else:
            #     has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num:', len(available_scenes))
    return available_scenes


def get_sample_data(nusc, sample_data_token, selected_anntokens=None):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    Args:
        nusc:
        sample_data_token: Sample_data token.
        selected_anntokens: If provided only return the selected annotation.

    Returns:

    """
    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        box.velocity = nusc.box_velocity(box.token)
        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def fill_trainval_infos(data_path, nusc, train_scenes, val_scenes, test=False, max_sweeps=10):
    train_nusc_infos = []
    val_nusc_infos = []
    progress_bar = tqdm.tqdm(total=len(nusc.sample), desc='create_info', dynamic_ncols=True)

    ref_chan = 'LIDAR_TOP'  # The radar channel from which we track back n sweeps to aggregate the point cloud.
    chan = 'LIDAR_TOP'  # The reference channel of the current sample_rec that the point clouds are mapped to.

    for index, sample in enumerate(nusc.sample):
        progress_bar.update()

        ref_sd_token = sample['data'][ref_chan]
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']
        ref_lidar_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token)
        ref_cam_front_token = sample['data']['CAM_FRONT']
        ref_cam_path, _, ref_cam_intrinsic = nusc.get_sample_data(ref_cam_front_token)

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)
        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)

        info = {
            'lidar_path': Path(ref_lidar_path).relative_to(data_path).__str__(),
            'cam_front_path': Path(ref_cam_path).relative_to(data_path).__str__(),
            'cam_intrinsic': ref_cam_intrinsic,
            'token': sample['token'],
            'sweeps': [],
            'ref_from_car': ref_from_car,
            'car_from_global': car_from_global,
            'timestamp': ref_time,
        }

        sample_data_token = sample['data'][chan]
        curr_sd_rec = nusc.get('sample_data', sample_data_token)
        sweeps = []
        while len(sweeps) < max_sweeps - 1:
            if curr_sd_rec['prev'] == '':
                if len(sweeps) == 0:
                    sweep = {
                        'lidar_path': Path(ref_lidar_path).relative_to(data_path).__str__(),
                        'sample_data_token': curr_sd_rec['token'],
                        'transform_matrix': None,
                        'time_lag': curr_sd_rec['timestamp'] * 0,
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                curr_sd_rec = nusc.get('sample_data', curr_sd_rec['prev'])

                # Get past pose
                current_pose_rec = nusc.get('ego_pose', curr_sd_rec['ego_pose_token'])
                global_from_car = transform_matrix(current_pose_rec['translation'], Quaternion(current_pose_rec['rotation']), inverse=False)

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = nusc.get('calibrated_sensor', curr_sd_rec['calibrated_sensor_token'])
                car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']), inverse=False)

                tm = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
                lidar_path = nusc.get_sample_data_path(curr_sd_rec['token'])
                time_lag = ref_time - 1e-6 * curr_sd_rec['timestamp']
                sweep = {
                    'lidar_path': Path(lidar_path).relative_to(data_path).__str__(),
                    'sample_data_token': curr_sd_rec['token'],
                    'transform_matrix': tm,
                    'global_from_car': global_from_car,
                    'car_from_current': car_from_current,
                    'time_lag': time_lag,
                }
                sweeps.append(sweep)

        info['sweeps'] = sweeps

        assert len(info['sweeps']) == max_sweeps - 1, \
            f"sweep {curr_sd_rec['token']} only has {len(info['sweeps'])} sweeps, " \
            f"you should duplicate to sweep num {max_sweeps - 1}"

        if not test:
            annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]

            # the filtering gives 0.5~1 map improvement
            num_lidar_pts = np.array([anno['num_lidar_pts'] for anno in annotations])
            num_radar_pts = np.array([anno['num_radar_pts'] for anno in annotations])
            mask = (num_lidar_pts + num_radar_pts > 0)

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)[:, [1, 0, 2]]  # wlh == > dxdydz (lwh)
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(-1, 1)
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes])
            gt_boxes = np.concatenate([locs, dims, rots, velocity[:, :2]], axis=1)

            assert len(annotations) == len(gt_boxes) == len(velocity)

            info['gt_boxes'] = gt_boxes[mask, :]
            info['gt_boxes_velocity'] = velocity[mask, :]
            info['gt_names'] = np.array([map_name_from_general_to_detection[name] for name in names])[mask]
            info['gt_boxes_token'] = tokens[mask]
            info['num_lidar_pts'] = num_lidar_pts[mask]
            info['num_radar_pts'] = num_radar_pts[mask]

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    progress_bar.close()
    return train_nusc_infos, val_nusc_infos


def _fill_trainval_infos(data_path,nusc,nusc_can,
                         train_scenes,
                         val_scenes,
                         radar_version,
                         future_tokens,
                         filter_version,
                         test=False,
                         max_sweeps=6):
    from pyquaternion import Quaternion
    train_nusc_infos = []
    val_nusc_infos = []
    total_samples = _get_available_samples(nusc)
    progress_bar = tqdm.tqdm(total=len(total_samples), desc='create_info', dynamic_ncols=True)
    if max_sweeps > 13:
        raise ValueError(f'The number of sweeps are must be lower then 13')
    # from second.data.roiaware_pool3d import roiaware_pool3d_utils
    for index, sample in enumerate(total_samples):
        progress_bar.update()
        lidar_token = sample["data"]["LIDAR_TOP"]
        cam_front_token = sample["data"]["CAM_FRONT"]
        radar_front_token = sample['data']['RADAR_FRONT']
        sd_rec = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
        cs_record = nusc.get('calibrated_sensor',sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = get_sample_data(nusc, lidar_token)
        ####Change 0830
        # _, boxses_2,_ = nusc.get_sample_data(lidar_token)
        
        cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_front_token)
        radar_path,_,_ = nusc.get_sample_data(radar_front_token)
        ref_from_car = transform_matrix(cs_record['translation'], Quaternion(cs_record['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(pose_record['translation'], Quaternion(pose_record['rotation']), inverse=True)
        
        ref_radar_path_split = radar_path.split('/')
        ref_radar_path_split[-2] = 'SECOND_data/RADAR_'+filter_version+ '_'+radar_version+ '_'+ str(max_sweeps) + 'Sweeps'
        ref_radar_path_split[-1] = ref_radar_path_split[-1].split('.')[0] + '.npy'
        ref_radar_path_sweep_version = '/'.join(ref_radar_path_split)
        dir_data = '/'.join(ref_radar_path_sweep_version.split('/')[:-1])
        if not os.path.exists(dir_data):
            os.mkdir(dir_data)
        if filter_version == 'Valid_filter' and radar_version in ['vel_rel','vel_relCego','vel_relego','vel_abs']:
            try:
                radar_npy = np.load(ref_radar_path_sweep_version)
            except:
                ValueError('There is No point')
        else:
            radar_npy = make_multisweep_radar_data(nusc,nusc_can,sample,radar_version,max_sweeps,future_tokens,data_path)
            np.save(ref_radar_path_sweep_version, radar_npy)
        
        assert Path(lidar_path).exists(), (
            "you must download all trainval data, key-frame only dataset performs far worse than sweeps."
        )
        info = {
            'lidar_path': Path(lidar_path).relative_to(data_path).__str__(),
            'cam_front_path': Path(cam_path).relative_to(data_path).__str__(),
            'radar_front_path': Path(radar_path).relative_to(data_path).__str__(),
            "radar_path": ref_radar_path_sweep_version,
            "token": sample["token"],
            "sweeps": [],
            'ref_from_car': ref_from_car,
            'car_from_global': car_from_global,
            "timestamp": sample["timestamp"],
        }

        l2e_r = cs_record['rotation']
        l2e_t = cs_record['translation']
        e2g_r = pose_record['rotation']
        e2g_t = pose_record['translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes if b.name in map_name_from_general_to_detection]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes if b.name in map_name_from_general_to_detection]).reshape(-1, 3)[:, [1, 0, 2]]
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes if b.name in map_name_from_general_to_detection]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]
            tokens = np.array([b.token for b in boxes])
            names = np.array([b.name for b in  boxes])
            names = np.array([map_name_from_general_to_detection[name] for name in names if name in map_name_from_general_to_detection])
            # import pdb; pdb.set_trace()
            # we need to convert rot to SECOND format.
            # change the rot format will break all checkpoint, so...
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)           
            assert(len(gt_boxes) == len(names))
            # for make new multisweep BEV GT boxes
            ref_boxes = copy.deepcopy(gt_boxes)
            ref_boxes[:,2]=0
            ref_boxes[:,5]=10
            points = radar_npy
            
            ####check_0830____
            gt_boxes_2 = np.concatenate([locs, dims[:,[1,0,2]], -rots - np.pi / 2], axis=1)
            ref_boxes_2 = copy.deepcopy(gt_boxes_2)
            ref_boxes_2[:,2]=0
            ref_boxes_2[:,5]=10

            indices_3 = points_count_rbbox(points,ref_boxes_2)
            mask_3 = np.array([k for k,v in enumerate(indices_3) if v >0])
            radar_points_3 = np.array([v for k,v in enumerate(indices_3) if v >0])
            
            # To find the gt box that include multisweep point
            ####change 0831
            if len(gt_boxes) != len(names):
                import pdb; pdb.set_trace()
            
            if mask_3.dtype != 'int' and mask_3.dtype != 'bool':
                mask_3 =  np.array([a["num_lidar_pts"]==-1 for a in annotations])
            # assert len(gt_boxes) == len(annotations), f"{len(gt_boxes)}, {len(annotations)}"
            try:
                info["gt_boxes"] = gt_boxes[mask_3]
                info["gt_names"] = names[mask_3]
                info['gt_second_boxes'] = gt_boxes_2[mask_3]
                info["gt_velocity"] = velocity.reshape(-1, 2)[mask_3]
                info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])[mask_3]
                info["num_radar_pts"] = radar_points_3
            except:
                new_mask = radar_points_3>0
                info["gt_boxes"] = gt_boxes[new_mask]
                info["gt_names"] = names[new_mask]
                info['gt_second_boxes'] = gt_boxes_2[new_mask]
                info["gt_velocity"] = velocity.reshape(-1, 2)[new_mask]
                info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])[new_mask]
                info["num_radar_pts"] = radar_points_3

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)
    progress_bar.close()
    return train_nusc_infos, val_nusc_infos

def boxes_lidar_to_nusenes(det_info,eval_x20):
    boxes3d = det_info['boxes_lidar']
    scores = det_info['score']
    labels = det_info['pred_labels']
    if eval_x20:
        box_x_mask = np.where(np.abs(boxes3d[:,0])<20)
        box_y_mask = np.where(np.abs(boxes3d[:,1])<50)
        box_mask = np.intersect1d(box_x_mask ,box_y_mask )
        boxes3d= boxes3d[box_mask]
        scores = scores[box_mask]
        labels = labels[box_mask]
    box_list = []
    for k in range(boxes3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=boxes3d[k, 6])
        velocity = (*boxes3d[k, 7:9], 0.0) if boxes3d.shape[1] == 9 else (0.0, 0.0, 0.0)
        box = Box(
            boxes3d[k, :3],
            boxes3d[k, [4, 3, 5]],  # wlh
            quat, label=labels[k], score=scores[k], velocity=velocity,
        )
        box_list.append(box)
    return box_list

def lidar_nusc_box_to_global(nusc, boxes, sample_token):
    s_record = nusc.get('sample', sample_token)
    sample_data_token = s_record['data']['LIDAR_TOP']

    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))
        # Move box to global coord system
        box.rotate(Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))
        box_list.append(box)
    return box_list


def transform_det_annos_to_nusc_annos(det_annos, nusc, eval_x20):
    nusc_annos = {
        'results': {},
        'meta': None,
    }

    for det in det_annos:
        annos = []
        box_list = boxes_lidar_to_nusenes(det,eval_x20)
        box_list = lidar_nusc_box_to_global(
            nusc=nusc, boxes=box_list, sample_token=det['metadata']['token']
        )
        # import pdb ;pdb.set_trace()
        for k, box in enumerate(box_list):
            name = det['name'][k]
            if name not in ['car', 'bicycle', 'bus', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', 'barrier']:
                continue
            if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
                    attr = 'vehicle.moving'
                elif name in ['bicycle', 'motorcycle']:
                    attr = 'cycle.with_rider'
                else:
                    attr = None
            else:
                if name in ['pedestrian']:
                    attr = 'pedestrian.standing'
                elif name in ['bus']:
                    attr = 'vehicle.stopped'
                else:
                    attr = None
            attr = attr if attr is not None else max(
                cls_attr_dist[name].items(), key=operator.itemgetter(1))[0]
            nusc_anno = {
                'sample_token': det['metadata']['token'],
                'translation': box.center.tolist(),
                'size': box.wlh.tolist(),
                'rotation': box.orientation.elements.tolist(),
                'velocity': box.velocity[:2].tolist(),
                'detection_name': name,
                'detection_score': box.score,
                'attribute_name': attr
            }
            annos.append(nusc_anno)

        nusc_annos['results'].update({det["metadata"]["token"]: annos})

    return nusc_annos


def make_multisweep_radar_data(nusc, nusc_can, sample, radar_version, max_sweeps, future_tokens, root_path):
    from nuscenes.utils.data_classes import RadarPointCloud
    from pyquaternion import Quaternion
    from nuscenes.utils.geometry_utils import transform_matrix
    all_pc = np.zeros((0, 18))

    # lidar information at sample time
    lidar_token = sample["data"]["LIDAR_TOP"]
    ref_sd_rec = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
    ref_cs_record = nusc.get('calibrated_sensor',ref_sd_rec['calibrated_sensor_token'])
    ref_pose_record = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    
    # For using can_bus data
    scene_rec = nusc.get('scene',sample['scene_token'])
    scene_name = scene_rec['name']
    scene_pose = nusc_can.get_messages(scene_name,'pose')
    # utimes, vel_data and yaw_rate are all data types in NumPy array format. 
    utimes = np.array([m['utime'] for m in scene_pose])
    vel_data = np.array([m['vel'][0] for m in scene_pose])
    yaw_rate = np.array([m['rotation_rate'][2] for m in scene_pose])

    # lidar transformation matrix information
    l2e_r = ref_cs_record['rotation']
    l2e_t = ref_cs_record['translation']
    e2g_r = ref_pose_record['rotation']
    e2g_t = ref_pose_record['translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    # Directions of radar data
    radar_channels = ['RADAR_FRONT','RADAR_FRONT_RIGHT','RADAR_FRONT_LEFT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

    for radar_channel in radar_channels:
        # each radar data information
        radar_data_token = sample['data'][radar_channel]
        radar_sample_data = nusc.get('sample_data',radar_data_token)
        ref_radar_time = radar_sample_data['timestamp']

        # At each sweep, We need to merge data from 5 directions
        for _ in range(max_sweeps):
            radar_path = osp.join(root_path,radar_sample_data['filename'])
            radar_cs_record = nusc.get('calibrated_sensor',radar_sample_data['calibrated_sensor_token'])
            radar_pose_record = nusc.get('ego_pose', radar_sample_data['ego_pose_token'])
            time_gap = (ref_radar_time-radar_sample_data['timestamp'])*1e-6

            #Use can_bus data to get ego_vehicle_speed, We get the Ego vehicle speed that fits the radar sensor data
            time_idx = np.argmin(np.abs(utimes-ref_radar_time))
            current_from_car = transform_matrix([0,0,0], Quaternion(radar_cs_record['rotation']),inverse=True)[0:2,0:2]
            # vel_data = scalar data, vehicle_v = vector that coordinate is ego, final_v = vector that coordinate is radar sensor
            vel_data[time_idx] = np.around(vel_data[time_idx],10)
            vehicle_v = np.array((vel_data[time_idx]*np.cos(yaw_rate[time_idx]),vel_data[time_idx]*np.sin(yaw_rate[time_idx])))          
            final_v = vehicle_v@current_from_car.T 

            # get radar data and do some filtering
            current_pc = RadarPointCloud.from_file(radar_path).points.T
            current_pc = point_filtering(current_pc, filter_version='Valid_filter')
            if current_pc == 'No_point':
                continue
            # version2 == Rel / absolute / absolute + Ego vehicle
            current_pc[:,:2] += current_pc[:,8:10]*time_gap

            ## make accurate Absolute velocity
            if radar_version == 'vel_relego':
                current_pc[:,6:8] = current_pc[:,8:10] + final_v[0:2]
            elif radar_version in ['vel_rel','vel_relCego']:
                current_pc[:,6:8] = current_pc[:,8:10]
            elif radar_version in ['vel_relCabsCego']:
                current_pc[:,16:18] = current_pc[:,8:10] + final_v[0:2]
            else:
                raise NotImplementedError 

            #radar_point to lidar top coordinate
            r2e_r_s = radar_cs_record['rotation']
            r2e_t_s = radar_cs_record['translation']
            e2g_r_s = radar_pose_record['rotation']
            e2g_t_s = radar_pose_record['translation']
            r2e_r_s_mat = Quaternion(r2e_r_s).rotation_matrix
            e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

            R = (r2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
            T = (r2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
            T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T

            current_pc[:,:3] = current_pc[:,:3] @ R
            current_pc[:,[6,7]] = current_pc[:,[6,7]] @ R[:2,:2]
            if radar_version == 'vel_relCabsCego':
                current_pc[:,[16,17]] = current_pc[:,[16,17]] @ R[:2,:2]
            current_pc[:,:3] += T            
            current_pc[:,[8,9]] = final_v[0:2] @ R[:2,:2]
            all_pc = np.vstack((all_pc, current_pc))

            if sample['token'] in future_tokens:
                if radar_sample_data['next'] == '':
                    break
                else:
                    radar_sample_data = nusc.get('sample_data', radar_sample_data['next'])                
            else:
                if radar_sample_data['prev'] == '':
                    break
                else:
                    radar_sample_data = nusc.get('sample_data', radar_sample_data['prev'])
    filter_version = 'Valid_filter'
    if radar_version in ['vel_rel','vel_relego']:
        if filter_version == 'No_filter_V2':
            use_idx = [0,1,2,5,6,7,3]
        else:
            use_idx = [0,1,2,5,6,7]
    elif radar_version == 'vel_relCego':
        if filter_version == 'No_filter_V2':
            use_idx = [0,1,2,5,6,7,8,9,3]
        else:
            use_idx = [0,1,2,5,6,7,8,9]
    ## New radar!! 0805
    elif radar_version == 'vel_relCabsCego':
        use_idx = [0,1,2,5,6,7,16,17,8,9]
    else:
        raise NotImplementedError
    all_pc = all_pc[:,use_idx]
    return all_pc

def point_filtering(pc, filter_version):
    if filter_version == 'No_filter_V1':
        return pc
    elif filter_version in ['Valid_filter','Full_filter','No_filter_V2']:
        ambig = pc[:,11]
        invalid = pc[:,14]
        dyn = pc[:,3]
        ambig_list = np.where(ambig==3)[0]
        if filter_version in ['Valid_filter','No_filter_V2']:
            valid_criteria = [0,4,8,9,10,11,12,15,16]
            invalid_list = np.array([idx for idx,point in enumerate(invalid) if point in valid_criteria])
            intersect = np.intersect1d(ambig_list,invalid_list)
            if filter_version == 'No_filter_V2':
                relative = np.setdiff1d(np.arange(len(pc)),intersect)
        else:
            valid_list = np.where(valid==0)[0]
            dyn_criteria = list(range(7))
            dyn_list = np.array([idx for idx,point in enumerate(dyn) if point in dyn_criteria])
            intersect = np.intersect1d(ambig_list,dyn_list,valid_list)
        if len(intersect) == 0:
            return 'No_point'
        else:
            if filter_version == 'No_filter_V2':
                pc[intersect,3] == 1 # valid point
                pc[relative,3] == 0 # invalid point
                return pc
            else:
                return pc[intersect,:]
    else:
        raise ValueError


def _second_gt_to_nusc_box(info, eval_x20,nusc):
    Class = ['car', 'bicycle', 'bus', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', 'barrier']
    from nuscenes.utils.data_classes import Box
    import pyquaternion
    box3d = info['gt_second_boxes']
    labels = info['gt_names']
    sample_tk = info['token']

    s_record = nusc.get('sample', sample_tk)
    sample_data_token = s_record['data']['LIDAR_TOP']

    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    ## for changing gt box range
    if eval_x20:
        box_x_mask = np.where(np.abs(box3d[:,0])<20)
        box_y_mask = np.where(np.abs(box3d[:,1])<50)
        box_mask = np.intersect1d(box_x_mask ,box_y_mask )
        box3d= box3d[box_mask]
        labels = labels[box_mask]
    scores = -1*np.ones(len(box3d))
    box3d[:, 6] = -box3d[:, 6] - np.pi / 2
    box_list = []
    # import pdb; pdb.set_trace()
    for i in range(box3d.shape[0]):
        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.config import DetectionConfig
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box3d[i, 6])
        velocity = (np.nan, np.nan, np.nan)
        if labels[i] not in Class:
            continue
        else:
            label = Class.index(labels[i])
        if box3d.shape[1] == 9:
            velocity = (*box3d[i, 7:9], 0.0)
            # velo_val = np.linalg.norm(box3d[i, 7:9])
            # velo_ori = box3d[i, 6]
            # velocity = (velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = Box(
            box3d[i, :3],
            box3d[i, 3:6],
            quat,
            label=label,
            score=scores[i],
            velocity=velocity)
        box.rotate(Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))
        # Move box to global coord system
        cfg = config_factory("detection_cvpr_2019")
        cls_range_map = cfg.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[Class[box.label]]
        if radius > det_range:
            continue
        box.rotate(Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))
        box_list.append(box)
    return box_list


def ground_truth_bbox(infos, class_names,  eval_x20, nusc):
    if "gt_boxes" not in infos[0]:
        return None
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.config import DetectionConfig
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.data_classes import DetectionBox
    cfg = config_factory('detection_cvpr_2019')
    cls_range_map = cfg.class_range
    mapped_class_names = class_names
     
    gt_annos={}
    for info in infos:
        sample_boxes = []
        sample_token = info['token']
        annos=[]
        boxes = _second_gt_to_nusc_box(info,eval_x20,nusc)
        for idx, box in enumerate(boxes):
            name = info['gt_names'][idx]
            if name not in ['car', 'bicycle', 'bus', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', 'barrier']:
                continue
            if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
                    attr = 'vehicle.moving'
                elif name in ['bicycle', 'motorcycle']:
                    attr = 'cycle.with_rider'
                else:
                    attr = None
            else:
                if name in ['pedestrian']:
                    attr = 'pedestrian.standing'
                elif name in ['bus']:
                    attr = 'vehicle.stopped'
                else:
                    attr = None
            attr = attr if attr is not None else max(
                cls_attr_dist[name].items(), key=operator.itemgetter(1))[0]
            velocity = box.velocity[:2].tolist()
            radar_pts = int(info['num_radar_pts'][idx])
            nusc_anno={
                    'sample_token':sample_token,
                    'translation':box.center.tolist(),
                    'size':box.wlh.tolist(),
                    'rotation':box.orientation.elements.tolist(),
                    'velocity':velocity,
                    'num_pts':radar_pts,
                    'detection_name':name,
                    'detection_score':int(-1),  
                    'attribute_name':attr
                    }
            annos.append(nusc_anno)
        gt_annos[sample_token] = annos

    gt_annos_2 = {}
    for info in infos:
        sample_token = info['token']
        boxes = info['gt_boxes']
        names = info['gt_names']
        point = np.load(info['radar_path'])
        x_mask = np.where(np.abs(point[:,0])<50)
        y_mask = np.where(np.abs(point[:,1])<50)
        mask = np.intersect1d(x_mask,y_mask)
        box_x_mask = np.where(np.abs(boxes[:,0])<50)
        box_y_mask = np.where(np.abs(boxes[:,1])<50)
        box_mask = np.intersect1d(box_x_mask ,box_y_mask )
        points = point[mask]
        boxes = boxes[box_mask].tolist()
        names = names[box_mask].tolist()
        new_point = points[:,:2].tolist()
        annos_2 = {'gt_boxes': boxes,'gt_names': names, 'point': new_point}
        gt_annos_2[sample_token] = annos_2
    
    return gt_annos, gt_annos_2


def format_nuscene_results(metrics, class_names, version='default'):
    result = '----------------Nuscene %s results-----------------\n' % version
    for name in class_names:
        threshs = ', '.join(list(metrics['label_aps'][name].keys()))
        ap_list = list(metrics['label_aps'][name].values())

        err_name =', '.join([x.split('_')[0] for x in list(metrics['label_tp_errors'][name].keys())])
        error_list = list(metrics['label_tp_errors'][name].values())

        result += f'***{name} error@{err_name} | AP@{threshs}\n'
        result += ', '.join(['%.2f' % x for x in error_list]) + ' | '
        result += ', '.join(['%.2f' % (x * 100) for x in ap_list])
        result += f" | mean AP: {metrics['mean_dist_aps'][name]}"
        result += '\n'

    result += '--------------average performance-------------\n'
    details = {}
    for key, val in metrics['tp_errors'].items():
        result += '%s:\t %.4f\n' % (key, val)
        details[key] = val

    result += 'mAP:\t %.4f\n' % metrics['mean_ap']
    result += 'NDS:\t %.4f\n' % metrics['nd_score']

    details.update({
        'mAP': metrics['mean_ap'],
        'NDS': metrics['nd_score'],
    })

    return result, details
