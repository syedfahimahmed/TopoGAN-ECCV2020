import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

def rescale(img,min_v=0,max_v=1):
    if np.max(img) - np.min(img) == 0:
        return np.zeros(img.shape,dtype=np.float32)
    rescaled= (img - np.min(img)) / (np.max(img) - np.min(img))
    return rescaled * (max_v - min_v) + min_v

def standarize(img):
    return (img - np.mean(img)) / np.std(img)

def open_grey_img(file):
    return np.array(Image.open(file).convert('L')).astype(np.float32)

def read_params(file):
    params = []
    with open(file, 'r') as infile:
        for line in infile:
            line = line.strip().split(',')
            numbers = list(map(float, line))
            params.append(numbers)
    return np.array(params).astype(np.float32)

def read_index_file(file):
    index_list = []
    with open(file,'r') as infile:
        for line in infile:
            index_list.append(line.strip())
    
    return index_list

def load_np_file(filename):
    return np.load(filename)

def load_image_file(filename):
    image = load_np_file(filename).astype(np.float32)
    image = rescale(image.transpose(2,1,0))
    return image

def load_boundary_file(filename):
    boundary = load_np_file(filename).astype(np.float32)
    # w,h = boundary.shape[0], boundary.shape[1]
    # new_b = boundary.copy()
    # for x in np.arange(w):
    #     for y in np.arange(h):
    #         if (len(boundary.shape) == 2):
    #             if boundary[x,y] == 1:
    #                 if (x > 0):
    #                     if (y > 0):
    #                         new_b[x-1,y-1] = 1;
    #                     new_b[x-1,y] = 1;
    #                     if (y < h - 1):
    #                         new_b[x-1,y+1] = 1;
                    
    #                 if (y > 0):
    #                     new_b[x,y-1] = 1;
    #                 if (y < h - 1):
    #                     new_b[x,y+1] = 1;

    #                 if (x < w - 1):
    #                     if (y > 0):
    #                         new_b[x+1,y-1] = 1;
    #                     new_b[x+1,y] = 1;
    #                     if (y < h - 1):
    #                         new_b[x+1,y+1] = 1;
    #         else:
    #             if (x > 0):
    #                 if (y > 0):
    #                     new_b[x-1,y-1,0] = 1;
    #                 new_b[x-1,y,0] = 1,0
    #                 if (y < h - 1):
    #                     new_b[x-1,y+1,0] = 1;
                
    #             if (y > 0):
    #                 new_b[x,y-1,0] = 1;
    #             if (y < h - 1):
    #                 new_b[x,y+1,0] = 1;

    #             if (x < w - 1):
    #                 if (y > 0):
    #                     new_b[x+1,y-1,0] = 1;
    #                 new_b[x+1,y,0] = 1;
    #                 if (y < h - 1):
    #                     new_b[x+1,y+1,0] = 1;
    new_b = boundary
    if len(boundary.shape) == 2:
        w,h = new_b.shape
        new_b = new_b.reshape([w,h,1])
    return new_b.transpose(2,1,0)


class BoundaryMaps(Dataset):
    def __init__(self, dir, channels_multiplier=1):
        self.boundary_dir = dir + "boundaries/"
        self.boundary_list = read_index_file(self.boundary_dir+'index.txt')
        
        self.n_imgs = len(self.boundary_list)
    
    def __getitem__(self, index):
        boundary = load_boundary_file(self.boundary_dir + self.boundary_list[index])
        
        return boundary

    def __len__(self):
        return self.n_imgs

class BoundaryDistance(Dataset):
    def __init__(self, dir, channels_multiplier=1):
        self.boundary_dir = dir + "boundaries/"
        self.boundary_list = read_index_file(self.boundary_dir+'index.txt')

        self.distance_dir = dir + "distances/"
        self.distance_list = read_index_file(self.distance_dir+'index.txt')
        
        self.n_imgs = len(self.boundary_list)
    
    def __getitem__(self, index):
        boundary = load_boundary_file(self.boundary_dir + self.boundary_list[index])

        dist_field = load_np_file(self.distance_dir + self.distance_list[index]).astype(np.float32)
        if (len(dist_field.shape)==2):
            w,h = dist_field.shape
            dist_field = dist_field.reshape([w,h,1])
        dist_field = dist_field.transpose(2,1,0)
        return boundary, dist_field
    def __len__(self):
        return self.n_imgs

class ImageBoundary(Dataset):
    def __init__(self, dir, channels_multiplier=1):
        self.img_dir = dir + "images/"
        self.img_list = read_index_file(self.img_dir+'index.txt')

        self.boundary_dir = dir + "boundaries/"
        self.boundary_list = read_index_file(self.boundary_dir+'index.txt')
        
        self.n_imgs = len(self.img_list)
    
    def __getitem__(self, index):
        image = load_np_file(self.img_dir + self.img_list[index]).astype(np.float32)
        image = rescale(image.transpose(2,1,0),-1,1)

        boundary = load_boundary_file(self.boundary_dir + self.boundary_list[index]).astype(np.float32)
        boundary=rescale(boundary,-1,1)
        return image, boundary
    def __len__(self):
        return self.n_imgs

class ImageBoundaryDistance(Dataset):
    def __init__(self, dir, channels_multiplier=1):
        self.img_dir = dir + "images/"
        self.img_list = read_index_file(self.img_dir+'index.txt')

        self.boundary_dir = dir + "boundaries/"
        self.boundary_list = read_index_file(self.boundary_dir+'index.txt')

        self.distance_dir = dir + "distances/"
        self.distance_list = read_index_file(self.distance_dir+'index.txt')
        
        self.n_imgs = len(self.img_list)
    
    def __getitem__(self, index):
        image  = load_image_file(self.img_dir + self.img_list[index])

        boundary = load_boundary_file(self.boundary_dir + self.boundary_list[index])

        dist_field = load_np_file(self.distance_dir + self.distance_list[index]).astype(np.float32)
        if (len(dist_field.shape)==2):
            w,h = dist_field.shape
            dist_field = dist_field.reshape([w,h,1])
        dist_field = dist_field.transpose(2,1,0)

        return image, boundary, dist_field

    def __len__(self):
        return self.n_imgs

class ImageDistance(Dataset):
    def __init__(self, dir, channels_multiplier=1):
        self.img_dir = dir + "images/"
        self.img_list = read_index_file(self.img_dir+'index.txt')


        self.distance_dir = dir + "distances/"
        self.distance_list = read_index_file(self.distance_dir+'index.txt')
        
        self.n_imgs = len(self.img_list)
    
    def __getitem__(self, index):
        image  = load_image_file(self.img_dir + self.img_list[index])

        dist_field = load_np_file(self.distance_dir + self.distance_list[index]).astype(np.float32)
        if (len(dist_field.shape)==2):
            w,h = dist_field.shape
            dist_field = dist_field.reshape([w,h,1])
        dist_field = dist_field.transpose(2,1,0)

        return image, dist_field

    def __len__(self):
        return self.n_imgs