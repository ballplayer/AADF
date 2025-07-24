import os
import torch

from attack_dateset import fb_attack_dataset, art_attack_dataset, new_white, new_black, new_physical_world, \
     od_attack, seg_attack
from audio_classification import audio_attack
from text_classification import text_attack
from attack_init import dataset_init, model_init, fb_init, art_init


def attacks(task, model_name, dataset_name, method_name, epsilons, save_path, save_boxes_path=None):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    distance_str = 'l2'
    targeted = False
    target_class = None
    steps = None
    max_iter = None

    if task == 0:
        data_loader, batch_size, shape, clip_values, preprocessing, nb_classes, y_target = \
            dataset_init(dataset_name, model_name, method_name, targeted, target_class)
        model = model_init(model_name, dataset_name, nb_classes, device)
    elif task == 1:
        od_attack(dataset_name, model_name, method_name, save_path, save_boxes_path)
        return
    elif task == 2:
        substitute = True
        if model_name == 'transform':
            substitute = False
        seg_attack(dataset_name, model_name, method_name, save_path, save_boxes_path, epsilons, device, substitute)
        return
    elif task == 3:
        audio_attack(dataset_name, model_name, save_path, save_boxes_path, epsilons, device)
        return
    elif task == 4:
        text_attack(dataset_name, model_name, save_path)
        return

    if method_name in ['VirtualAdversarialAttack', 'BinarySearchContrastReductionAttack', 'BinarizationRefinementAttack',
                       'DatasetAttack', 'GaussianBlurAttack', 'LinearSearchBlendedUniformNoiseAttack',
                       'LinearSearchContrastReductionAttack', 'BoundaryAttack', 'L2CarliniWagnerAttack',
                       'DDNAttack', 'EADAttack', 'LinfFastGradientAttack', 'NewtonFoolAttack', 'L1FMNAttack',
                       'LinfProjectedGradientDescentAttack', 'SaltAndPepperNoiseAttack', 'L1BrendelBethgeAttack',
                       'L2AdditiveGaussianNoiseAttack', 'L2AdditiveUniformNoiseAttack', 'L2BasicIterativeAttack',
                       'L2BrendelBethgeAttack', 'L2ClippingAwareAdditiveGaussianNoiseAttack', 'LInfFMNAttack',
                       'L2ClippingAwareAdditiveUniformNoiseAttack', 'L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack',
                       'L2ClippingAwareRepeatedAdditiveUniformNoiseAttack', 'L2ContrastReductionAttack',
                       'L2DeepFoolAttack', 'L2FastGradientAttack', 'L2FMNAttack', 'L2ProjectedGradientDescentAttack',
                       'L2RepeatedAdditiveGaussianNoiseAttack', 'L2RepeatedAdditiveUniformNoiseAttack',
                       'LinfAdditiveUniformNoiseAttack', 'LinfBasicIterativeAttack', 'LinfDeepFoolAttack',
                       'LinfinityBrendelBethgeAttack', 'LinfRepeatedAdditiveUniformNoiseAttack']:
        if dataset_name in ['MSTAR', 'ImageNet']:
            if method_name in ['L1BrendelBethgeAttack', 'L2BrendelBethgeAttack', 'LinfinityBrendelBethgeAttack',
                               'VirtualAdversarialAttack']:
                steps = 200
            elif method_name in ['EADAttack', 'LinearSearchContrastReductionAttack', 'SaltAndPepperNoiseAttack',
                                 'BoundaryAttack', 'GaussianBlurAttack', 'LinearSearchBlendedUniformNoiseAttack',
                                 'L2CarliniWagnerAttack']:
                steps = 500
        ae_model, attack = fb_init(model, clip_values, method_name, preprocessing, steps, distance_str, device)
        fb_attack_dataset(attack, model_name, data_loader, dataset_name, save_path, method_name, ae_model, device,
                          epsilons)

    elif method_name in ['AutoAttack', 'AutoProjectedGradientDescent', 'AutoConjugateGradient',
                         'CarliniL0Method', 'CarliniLInfMethod', 'SaliencyMapMethod', 'GeoDA', 'SquareAttack']:
        ae_model = art_init(model, nb_classes, method_name, shape, clip_values, preprocessing, device)

        if dataset_name in ['MSTAR', 'ImageNet']:
            if method_name == 'GeoDA':
                max_iter = 500
            # elif method_name in ['AutoAttack', 'AutoProjectedGradientDescent']:
            #     eps_step = 0.01

        art_attack_dataset(model_name, data_loader, batch_size, dataset_name, save_path, method_name, ae_model,
                           epsilons, targeted, y_target, max_iter)

    # 新研白盒
    elif method_name in ['L2UMN', 'L2GMN', 'LinfUMN', 'L2CGMN', 'L2CUMN']:
        if method_name == 'L2UMN':
            first_method = 'L2AdditiveUniformNoiseAttack'
        elif method_name == 'L2GMN':
            first_method = 'L2AdditiveGaussianNoiseAttack'
        elif method_name == 'LinfUMN':
            first_method = 'LinfAdditiveUniformNoiseAttack'
        elif method_name == 'L2CGMN':
            first_method = 'L2ClippingAwareAdditiveGaussianNoiseAttack'
        else:
            first_method = 'L2ClippingAwareAdditiveUniformNoiseAttack'

        ae_model, attack = fb_init(model, clip_values, first_method, preprocessing, None, None, device)

        ae_model2, attack2 = fb_init(model, clip_values, 'LInfFMNAttack', preprocessing, None, None, device)

        new_white(attack, model_name, data_loader, dataset_name, save_path, method_name, ae_model, device, epsilons,
                  ae_model2, attack2)

    # 新研黑盒
    elif method_name in ['L2UGD', 'L2GGD', 'LinfUGD', 'L2CGGD', 'L2CUGD']:
        if method_name == 'L2UGD':
            first_method = 'L2AdditiveUniformNoiseAttack'
        elif method_name == 'L2GGD':
            first_method = 'L2AdditiveGaussianNoiseAttack'
        elif method_name == 'LinfUGD':
            first_method = 'LinfAdditiveUniformNoiseAttack'
        elif method_name == 'L2CGGD':
            first_method = 'L2ClippingAwareAdditiveGaussianNoiseAttack'
        else:
            first_method = 'L2ClippingAwareAdditiveUniformNoiseAttack'

        ae_model, attack = fb_init(model, clip_values, first_method, preprocessing, None, None, device)
        ae_model2 = art_init(model, nb_classes, 'GeoDA', shape, clip_values, preprocessing, device)
        new_black(attack, model_name, data_loader, dataset_name, save_path, method_name, ae_model, device, batch_size,
                  epsilons, ae_model2)

    # 新研物理世界
    elif method_name in ['GeneralAdversarialAttack', 'L2UAP']:
        if method_name == 'GeneralAdversarialAttack':
            first_method = 'L2RepeatedAdditiveUniformNoiseAttack'
        else:
            first_method = 'L2AdditiveUniformNoiseAttack'
        ae_model, attack = fb_init(model, clip_values, first_method, preprocessing, None, None, device)
        ae_model2 = art_init(model, nb_classes, 'AdversarialPatch', shape, clip_values,
                             preprocessing, device)
        new_physical_world(attack, model_name, data_loader, dataset_name, save_path, method_name, ae_model, device,
                           batch_size, epsilons, ae_model2, shape)


