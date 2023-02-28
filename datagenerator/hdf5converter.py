from paraview import simple as ps
from paraview import servermanager as sm
from datetime import datetime
from scipy import ndimage
import h5py
import os
import numpy as np
import vtk
import sys
import argparse
import gc

ps.LoadPlugin('C:/Program Files/ParaView 5.11.0/bin/paraview-5.11/plugins/TopologyToolKit/TopologyToolKit.dll', remote=False, ns=globals())

def rescale(array, min_v = 0, max_v = 1):
    if np.min(array) == np.max(array):
        return array * 0
    rescaled = (array - np.min(array))/(np.max(array)- np.min(array))
    return rescaled * (max_v - min_v) + min_v

def check_dir(path):
    if not (os.path.exists(path) and os.path.isdir(path)):
        os.makedirs(path)

def hdf2image(data, sf_name='Scalars_'):
    d,h,w = data.shape
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(w,h,d)
    scalars = vtk.vtkFloatArray()
    scalars.SetNumberOfComponents(1)
    scalars.SetNumberOfTuples(w*h*d)
    scalars.SetName(sf_name)
    np_volume = np.array(data).astype(np.float32)
    if not np_volume.flags.contiguous:
        np_volume = np.ascontiguousarray(np_volume)
    volume_flat = np.ravel(np_volume)
    if not volume_flat.flags.contiguous:
        volume_flat = np.ascontiguousarray(volume_flat)
    scalars.SetVoidArray(volume_flat, volume_flat.shape[0], 1)
    image_data.GetPointData().AddArray(scalars)
    image_data.GetPointData().SetActiveScalars(sf_name)
    return image_data

def register_data(image_data):
    # register image_data to server
    vti_data = ps.TrivialProducer(registrationName='vti_data')
    client = vti_data.GetClientSideObject()
    client.SetOutput(image_data)
    vti_data.UpdatePipeline(time=0.0)
    return vti_data

def persistence_diagram(data_proxy, sf_name='Scalars_'):
    ttk_pd = TTKPersistenceDiagram(registrationName='ttk_pd', Input=data_proxy, DebugLevel=0)
    ttk_pd.ScalarField = ['POINTS', sf_name]
    ttk_pd.InputOffsetField = ['POINTS', sf_name]
    ps.UpdatePipeline(time=0.0, proxy=ttk_pd)
    return ttk_pd

def persistence_threshold(data_proxy, low=None, high=None):
    threshold1 = ps.Threshold(registrationName='Threshold1', Input=data_proxy)
    threshold1.Scalars = ['CELLS', 'Persistence']
    thresh_data = sm.Fetch(data_proxy)
    p_range = thresh_data.GetCellData().GetAbstractArray("Persistence").GetRange()
    low = p_range[0] if low == None else max([low, p_range[0]])
    high = p_range[1] if high == None else min([high, p_range[1]])
    threshold1.LowerThreshold = low
    threshold1.UpperThreshold = high
    ps.UpdatePipeline(time=0.0, proxy=threshold1)
    return threshold1

def simplification(data_proxy, threshold, sf_name='Scalars_'):
    ttk_persistence_simplification = TTKTopologicalSimplification(registrationName='ttk_persistence_simplification', Domain=data_proxy,
            Constraints=threshold, DebugLevel=0)
    ttk_persistence_simplification.ScalarField = ['POINTS', sf_name]
    ttk_persistence_simplification.InputOffsetField = ['POINTS', sf_name]
    ttk_persistence_simplification.VertexIdentifierField = ['POINTS', 'CriticalType']
    ttk_persistence_simplification.NumericalPerturbation = 1
    ps.UpdatePipeline(time=0.0, proxy=ttk_persistence_simplification)
    return ttk_persistence_simplification

def morse_complex(data_proxy):
    ttk_msc = TTKMorseSmaleComplex(registrationName='ttk_msc', Input=data_proxy, DebugLevel=0)
    ttk_msc.CriticalPoints = 1
    ttk_msc.Ascending1Separatrices = 0
    ttk_msc.Descending1Separatrices = 1
    ttk_msc.SaddleConnectors = 0
    ttk_msc.Ascending2Separatrices = 0
    ttk_msc.Descending2Separatrices = 0
    ttk_msc.AscendingSegmentation = 1
    ttk_msc.DescendingSegmentation = 0
    ttk_msc.MorseSmaleComplexSegmentation = 0
    ps.UpdatePipeline(time=0.0, proxy=ttk_msc)
    ps.UpdatePipeline(time=0.0, proxy=ttk_msc)
    ps.UpdatePipeline(time=0.0, proxy=ttk_msc)
    ps.UpdatePipeline(time=0.0, proxy=ttk_msc)
    return ttk_msc

def resample_image(data_proxy, size=[64,64]):
    resample_to_image = ps.ResampleToImage(registrationName='resample', Input=data_proxy)
    resample_to_image.SamplingDimensions = [size[0], size[1], 1]
    ps.UpdatePipeline(time=0.0, proxy=resample_to_image)
    return resample_to_image

def extract_arcs(poly_data, scalar_size, image_size, origin):
    vtk_points = poly_data.GetPoints()
    n_cells = poly_data.GetNumberOfCells()
    arcs_image = np.zeros(image_size,dtype=np.float32)
    xratio = scalar_size[0]/image_size[0]
    yratio = scalar_size[1]/image_size[1]
    for cell_idx in range(n_cells):
        cell = poly_data.GetCell(cell_idx)
        n_pts = cell.GetNumberOfPoints()
        for vert_idx in range(n_pts):
            pid = cell.GetPointId(vert_idx)
            x,y,_ = np.round((np.array(vtk_points.GetPoint(pid),dtype=np.float32)-origin)/np.array([xratio,yratio,1],dtype=np.float32))
            x = np.clip(x,0,image_size[0]-1).astype(np.int32)
            y = np.clip(y,0,image_size[1]-1).astype(np.int32)
            arcs_image[x,y] = 1
    arcs_image.reshape([image_size[0], image_size[1], 1])
    return arcs_image


def save_simplified_image(image_data, sf_name, path):
    dims = image_data.GetDimensions()
    vtk_array = image_data.GetPointData().GetAbstractArray(sf_name)
    # assumes float array. change dtype if different. doesn't support bit array
    np_img = np.frombuffer(vtk_array, dtype=np.float32, count=dims[0]*dims[1]*dims[2]).reshape(dims).transpose([1,0,2])
    np.save(path, np_img)

def prepare_output_directories(output_folder):
    if output_folder[-1] != '/':
        output_folder += '/'
    images_folder= output_folder + 'originals/'
    simples_folder = output_folder + 'images/'
    arcs_folder = output_folder + 'arcs/'
    boundaries_folder = output_folder  + 'boundaries/'
    dt_folder = output_folder + "distances/"
    check_dir(output_folder)
    check_dir(images_folder)
    check_dir(simples_folder)
    check_dir(arcs_folder)
    check_dir(boundaries_folder)
    check_dir(dt_folder)
    return output_folder,images_folder,simples_folder,arcs_folder,boundaries_folder,dt_folder


def process_patch(data, sf_name,scalar_size,image_size, x, y, z, folders, n):
    _,images_folder,simples_folder,arcs_folder,boundaries_folder,dt_folder = folders
    # create vti
    np_slice = data[z,y:y+scalar_size[1],x:x+scalar_size[0]].reshape([1,scalar_size[1],scalar_size[0]])
    d,h,w = np_slice.shape
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(w,h,d)
    scalars = vtk.vtkFloatArray()
    scalars.SetNumberOfComponents(1)
    scalars.SetNumberOfTuples(w*h*d)
    scalars.SetName(sf_name)
    if not np_slice.flags.contiguous:
        np_slice = np.ascontiguousarray(np_slice)
    volume_flat = np.ravel(np_slice)
    if not volume_flat.flags.contiguous:
        volume_flat = np.ascontiguousarray(volume_flat)
    scalars.SetVoidArray(volume_flat, volume_flat.shape[0], 1)
    image_data.GetPointData().AddArray(scalars)
    image_data.GetPointData().SetActiveScalars(sf_name)
    # read slice of volume and send it to server
    # image_data = hdf2image(np_slice,sf_name)
    original_image = register_data(image_data)
    # resampled_proxy = resample_image(original_image,image_size)
    # persistence simplification
    pd = persistence_diagram(original_image, sf_name)
    # persistence_threshold(proxy,low,high) when low/high are None they will be set automatically.
    # choose low in paraview/ttk before running it on script. dynamic choosing is a hard problem in visualization.
    threshold = persistence_threshold(pd, 15.0) 

    try:
        simplified_image = simplification(original_image, threshold, sf_name)
    except:
        simplified_image = original_image
    # debug output of image as vti files
    
    # save the model input scalar field. This is supposed to be the ground truth scalar image
    save_simplified_image(sm.Fetch(simplified_image),sf_name,f'{simples_folder}simplified_{n:05d}.npy')

    # compute Morse complex
    mc = morse_complex(simplified_image)

    # save image files
    ps.SaveData(f'{images_folder}original_{n:05d}.vti',proxy=original_image, ChooseArraysToWrite=1,PointDataArrays=[sf_name],CellDataArrays=[])
    ps.SaveData(f'{images_folder}simplified_{n:05d}.vti',proxy=simplified_image, ChooseArraysToWrite=1,PointDataArrays=[sf_name,f'{sf_name}_Order'],CellDataArrays=[])
    # save arcs as vtp files
    poly_data = sm.Fetch(mc, idx=1)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(f'{arcs_folder}arcs_{n:05d}.vtp')
    writer.SetInputData(poly_data)
    writer.Write()

    # extract arcs as numpy arrays
    arcs_image = extract_arcs(poly_data,scalar_size,image_size, np.array([0,0,0],dtype=np.float32))
    np.save(f'{boundaries_folder}boundary_{n:05d}.npy', arcs_image)
    distance = np.copy(arcs_image)
    distance[distance == 1] = 2
    distance[distance == 0] = 1
    distance[distance == 2] = 0
    np.save(f'{dt_folder}distance_{n:05d}.npy', ndimage.distance_transform_edt(distance))
    


def distance_transform(image):
    new_image = image.copy()
    new_image[new_image==0] = 2
    new_image[new_image==1] = 0
    new_image[new_image==2] = 1
    return ndimage.distance_transform_edt(new_image)

def read_params(param_file):
    if os.path.exists(param_file):
        with open(param_file, 'r') as file:
            params = []
            for line in file:
                i,x,y,z = list(map(lambda x:int(x), line.strip().split(',')))
                params.append([i,x,y,z])
        return params
    return []

def get_coords(idx,d,h):
    z = idx % d
    y = idx // d % h
    x = idx // (d * h)
    return x,y,z

def run_ranges(data,scalar_size,image_size,n_images,output_dir, index):
    npdata,sf_name = data
    folders = prepare_output_directories(output_dir)
    output_folder,_,simplified_folder,_,boundaries_folder,dt_folder = folders
    d,h,w = npdata.shape
    ha = h-scalar_size[1]
    wa = w-scalar_size[0]
    bounds = d * ha * wa
    print(f'{npdata.shape}')

    sum_iter_time = 0

    param_file_name = f'{output_folder}params.csv'
    params = read_params(param_file_name)

    simplified_index = open(f'{simplified_folder}index.txt','a')
    boundaries_index = open(f'{boundaries_folder}index.txt','a')
    distance_index = open(f'{dt_folder}index.txt','a')
    params_file = open(param_file_name,'a')
    
    for count in range(n_images):
        # timer
        t_start = datetime.now()
        idx = np.random.randint(0,bounds)
        x,y,z = get_coords(idx,d,ha)
        while [index,x,y,z] in params:
            idx = np.random.randint(0,bounds)
            x,y,z = get_coords(idx,d,ha)
        # for zi in range(5):
        #     for yi in range(5):
        #         for xi in range(5):
        nx,ny,nz = x, y, z
        process_patch(npdata,sf_name,scalar_size,image_size,nx,ny,nz,folders,len(params))
        simplified_index.write(f'simplified_{len(params):05d}.npy\n')
        boundaries_index.write(f'boundary_{len(params):05d}.npy\n')
        distance_index.write(f'distance_{len(params):05d}.npy\n')
        params_file.write(f"{index},{nx},{ny},{nz}\n")
        params.append([index,nx,ny,nz])

        simplified_index.flush()
        boundaries_index.flush()
        distance_index.flush()
        params_file.flush()
        t_end = datetime.now()
        # time
        iter_time = (t_end - t_start).total_seconds()
        sum_iter_time += iter_time
        ave_iter_time = sum_iter_time/(count+1)
        sys.stdout.write(f'\rprocessed {(count+1)}/{n_images} data in {sum_iter_time:.03f} seconds at {ave_iter_time:.03f}s/item. est:{(n_images-(count+1))*ave_iter_time:.03f}s')
    # end of for loop
    print()
    simplified_index.close()
    boundaries_index.close()
    distance_index.close()
    params_file.close()
    

def build_index(folder):
    if folder[-1] != '/':
        folder+='/'
    files = os.listdir(folder)
    with open(f'{folder}index.txt','w') as index_file:
        for file in files:
            index_file.write(f'{file}\n')

def read_data(data_file):
    h5file = h5py.File(data_file,'r')
    npdata = np.array(h5file['volumes']['raw'], dtype=np.float32)
    h5file.close()
    return npdata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='hdf5 file')
    parser.add_argument('output',help='output directory')
    parser.add_argument('n_images', type=int)
    parser.add_argument('--sf_name', default='Scalars_')
    parser.add_argument('--image_size', nargs=2, type=int, default=[64,64])
    parser.add_argument('--scalar_size', nargs=2, type=int, default=[256,256])
    parser.add_argument('--index',default=0,type=int, help='index of the dataset 1:A 2:B 3:C')
    opt = parser.parse_args()

    sf_name=opt.sf_name
    scalar_size = list(opt.scalar_size)
    image_size = list(opt.image_size)
    num_images = opt.n_images
    out_dir = opt.output
    if out_dir[-1]!='/':
        out_dir += '/'

    # prepare metainformation of dataset
    data = [read_data(opt.input),sf_name]
    run_ranges(data, scalar_size, image_size, num_images , opt.output, opt.index)

if __name__=='__main__':
    main()
