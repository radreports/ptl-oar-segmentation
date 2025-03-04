import torch, warnings, nrrd
import numpy as np
from skimage import measure, morphology
import string, random, glob

ROIS = ["External", "GTVp", "LCTVn", "RCTVn", "Brainstem", "Esophagus",
        "Larynx", "Cricoid_P", "OpticChiasm", "Glnd_Lacrimal_L",
        "Glnd_Lacrimal_R", "Lens_L", "Lens_R", "Eye_L", "Eye_R",
        "Nrv_Optic_L", "Nrv_Optic_R", "Parotid_L", "Parotid_R",
        "SpinalCord", "Mandible_Bone", "Glnd_Submand_L",
        "Glnd_Submand_R", "Cochlea_L", "Cochlea_R", "Lips",
        "Spc_Retrophar_R", "Spc_Retrophar_L", "BrachialPlex_R",
        "BrachialPlex_L", "BRAIN", "OralCavity", "Musc_Constrict_I",
        "Musc_Constrict_S", "Musc_Constrict_M"]

custom_order = [11,14,17,10,24,2,12,23,4,7,18,9,16,25,1,20,6,3,8,19,5,13]

#################################

def getROIOrder(custom_order=custom_order, rois=ROIS):
    # ideally this ordering has to be consistent to make inference easy...
    if custom_order is None:
        order_dic = {roi:i for i, roi in enumerate(ROIS)}
    else:
        roi_order = [rois[i] for i in custom_order]
        order_dic = {roi:i for i, roi in enumerate(roi_order)}
    return order_dic

# first step get ROI order
def getHeaderData(folders, structures=True, roi_order=getROIOrder()):
    assert len(folders) > 1
    voxel_dic = {}
    img_dic = {}
    # com_dic = {}
    oars_used = list(roi_order.keys())
    for list_ in folders:
        for i, p in enumerate(list_):
            oar = p.split('/')[-1].partition('.')[0]
            if oar in oars_used:
                class_idx = roi_order[oar]
                header = nrrd.read_header(p)
                voxels = header["Voxels"]
                try:
                    data = voxel_dic[oar]
                    voxel_dic[oar] = (data + voxels)/2.
                    if i==0:
                        mean = img_dic["meanHU"]
                        std = img_dic["stdHU"]
                        img_dic = {"meanHU":(mean+header["meanHU"])/2.,
                                   "stdHU":(std+header["stdHU"])/2}

                # can do the same thing if com data was in mask...
                # data = com_dic[oar]
                # com_dic[oar] = (data + voxels)/2.
                except Exception:
                    voxel_dic[oar] = voxels
                    img_dic = {"meanHU":header["meanHU"],
                               "stdHU":header["stdHU"]}
            else:
                #warnings.warn(f"{oar} not in list of chosen ROIS. If this is a mistake please update roi_order.")
                pass
                # com_dic[oar] = com

    return {"VOXINFO":voxel_dic, "IMGINFO":img_dic}

def swi(image, net, n_classes):

    # in our case we're looking for a 5D tensor...
    if len(image.size()) < 5:
        image = image.unsqueeze(0)

    shape = image.size()
    warnings.warn(f'Image shape is {image.size()}')

    z= 64
    y= 192

    # Z ...
    start = [0, shape[2]//4, shape[2]-shape[2]//4 - z, shape[2]-z]
    end = [z, shape[2]//4+z, shape[2]-shape[2]//4, shape[2]]
    # Y
    start_y = [0, shape[3]-y, shape[3]//12, shape[3]//6, shape[3] - shape[3]//4 - y, shape[3] - shape[3]//6 - y, shape[1] - shape[1]//12 - y]
    end_y = [y, shape[3], shape[3]//12 + y, shape[3]//6 + y, shape[3] - shape[3]//4, shape[3] - shape[3]//6, shape[1] - shape[1]//12]
    # X
    start_x = [0, shape[4]-y, shape[4]//12, shape[4]//4, shape[4]//6, shape[4] - shape[4]//4 - y, shape[4] - shape[4]//6 - y, shape[4] - shape[4]//12 - y]
    end_x = [y, shape[4], shape[4]//12 + y, shape[4]//4+y, shape[4]//6 +y, shape[4] - shape[4]//4, shape[4] - shape[4]//6, shape[4] - shape[4]//12]
    output_shape = (1, n_classes, shape[2], shape[3], shape[4])

    reference_ = torch.zeros(output_shape).to(image.device)
    reference = torch.zeros(output_shape).to(image.device)
    iter_=0
    for i, val in enumerate(start):
        for j, v in enumerate(start_y):
            for k, va in enumerate(start_x):
                im = image[:,:,val:end[i], v:end_y[j], va:end_x[k]]
                sh = im.size() # shoud be 5D tensor...
                if (sh[1], sh[2], sh[3], sh[4])!= (1, z, y, y):
                    pass
                else:
                    reference_[:,:,val:end[i], v:end_y[j], va:end_x[k]]+=1
                    # RUN THE NETWORK
                    im=im[0]
                    warnings.warn(f'NET IN SIZE IS {im.size()}')
                    output=net(im)
                    
                    warnings.warn(f'NET OUTS SIZE IS {output.size()}')
                    reference[:,:, val:end[i], v:end_y[j], va:end_x[k]] += output[:,:-1,:,:,:]
                    iter_ += 1

    warnings.warn(f'Iterated {iter_} times with sliding window.')
    return reference/reference_

#takes in an image slice and returns a list of all contours indexed by the class labels
def get_contours(img, num_classes, level):
    contour_list = []
    for class_id in range(num_classes):
        out_copy = img.copy()
        out_copy[out_copy != class_id] = 0
        outed = out_copy
        contours = measure.find_contours(outed, level)
        contour_list.append(contours)
    return contour_list

#Create contour set for the entire volume: expected input in a volume of size (Z,Y,X)
#label_names are the names corresponding to each label index in model output ["background", "brain", "optic chiams", ...]
#                                                                               0            1           2       etc
#len(label_names) should equal num_classes
#Return format:
# {
#   "Brain":
#   [
#     [(x, y, z1), (x, y, z1), (x, y, z1), (x, y, z1)], # slice 1
#     [(x, y, z2), (x, y, z2), (x, y, z2), (x, y, z2)], # slice 2
#     [(x, y, z3), (x, y, z3), (x, y, z3), (x, y, z3)], # slice 3
#     .
#     .
#     .
#     [(x, y, zn), (x, y, zn), (x, y, zn), (x, y, zn)] # slice n
#   ]
# }
# Test notebook is on local

def numpy_to_contour(arr, num_classes, level, label_names):
    #create empty dict
    contour_set = {}
    for label in label_names:
        contour_set[label] = []

    #Go though each slice
    for z_value, arr_slice in enumerate(arr):
        contours_list = get_contours(arr_slice, num_classes, level) #get all contours for this slice

        #Go through each label's contours
        for label, site_contours in enumerate(contours_list):
            #Go through each contour in that specific label and append the z value
            for contour_id, contour in enumerate(site_contours):
                contours_list[label][contour_id] = np.insert(contours_list[label][contour_id], 2, z_value, axis=1) #add z value into contour coordinates

        #Append into the dictionary
        for i, label in enumerate(label_names): #for each organ
            for array in contours_list[i]:
                array = array.tolist() #convert from numpy array to python list
                contour_set[label].append(array)
                #contour_set[label] = np.append(contour_set[label], array)

    #can use json.dump() to save this as a json file
    return contour_set

##OLD
# def numpy_to_contour(arr, num_classes, level, label_names):
#
#     #create empty dict
#     contour_set = {}
#     for label in label_names:
#         contour_set[label] = []
#
#     #Go though each slice
#     for z_value, arr_slice in enumerate(arr):
#         contours_list = get_contours(arr_slice, num_classes, level) #get all contours for this slice
#
#         #Go through each label's contours
#         for label, site_contours in enumerate(contours_list):
#             #Go through each contour in that specific label and append the z value
#             for contour_id, contour in enumerate(site_contours):
#                 contours_list[label][contour_id] = np.insert(contours_list[label][contour_id], 2, z_value, axis=1) #add z value into contour coordinates
#
#         #Append into the dictionary
#         for i, label in enumerate(label_names):
#             for array in contours_list[i]:
#                 contour_set[label] = np.append(contour_set[label], array.flatten())
#
#     #can use json.dump() to save this as a json file
#     return contour_set
