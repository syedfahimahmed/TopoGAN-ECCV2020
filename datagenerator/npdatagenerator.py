from paraview import simple as ps
from paraview import servermanager as sm
from datetime import datetime
import h5py
import os
import numpy as np
import vtk
import sys
import argparse
import gc

from scipy import ndimage

ps.LoadPlugin('/home/jixianli/Applications/ParaView-5.11.0-RC2-MPI-Linux-Python3.9-x86_64/lib/paraview-5.11/plugins/TopologyToolKit/TopologyToolKit.so', remote=False, ns=globals())


def rescale(array, min_v=0, max_v=1):
    if np.min(array) == np.max(array):
        return array * 0
    rescaled = (array - np.min(array))/(np.max(array) - np.min(array))
    return rescaled * (max_v - min_v) + min_v


def check_dir(path):
    if not (os.path.exists(path) and os.path.isdir(path)):
        os.makedirs(path)


def get_scalar_function(size=[256, 256, 1], n_blobs=0, centers=[], sigma=[]):
    # abbr:sf -> scalar function
    x = np.arange(size[0])
    y = np.arange(size[1])
    xx, yy = np.meshgrid(x, y, indexing="xy")

    gs = np.zeros([n_blobs]+size)
    for i in range(n_blobs):
        gs[i, :, :, 0] = np.exp(-((xx - centers[i, 0])**2/(2 * sigma[i] ** 2) +
                                  (yy - centers[i, 1])**2/(2 * sigma[i] ** 2)))

    sf = np.sum(gs, axis=0)
    return rescale(sf)


def get_sine_scalar_function(size=[256, 256, 1], x_factor=1, y_factor=1, x_rot=1, y_rot=1, axis_aligned=False):
    # abbr:sf -> scalar function
    sf = np.zeros(size)
    for x in range(size[0]):
        for y in range(size[1]):
            if axis_aligned:
                sf[x][y] = np.sin(x*x_factor) + np.sin(y*y_factor)
            else:
                sf[x][y] = np.sin(x*x_factor + y/x_rot) + \
                    np.sin(y*y_factor + x/y_rot)
    return rescale(sf)


def gen_image(size=[256, 256, 1], axis_aligned=False):
    nf = noise(size, 0, 0.1)
    rint = np.random.randint(5, 20)
    roff = np.random.randint(5, 20)
    rint_rotx = np.random.randint(10, 90) * np.random.choice([-1, 1])
    rint_roty = np.random.randint(10, 90) * np.random.choice([-1, 1])

    sf = get_sine_scalar_function(size=size,
                                  x_factor=1 / rint,
                                  y_factor=1 / roff,
                                  x_rot=rint_rotx,
                                  y_rot=rint_roty,
                                  axis_aligned=axis_aligned)
    image = sf + nf
    return rescale(image)


def gen_blob(size=[256, 256, 1], n_min=5, n_max=32):
    # abbr:rn_blob -> random number of blobs
    rn_blobs = np.random.randint(n_min, n_max)
    # abbr:r_centers -> random centers
    r_centers = np.random.randint(0, size[0], [rn_blobs, 2])
    # abbr:r_sigs -> random sigmas
    r_sigs = np.random.uniform(4, 16, rn_blobs)

    sf = get_scalar_function(size=size, n_blobs=rn_blobs,
                             centers=r_centers, sigma=r_sigs)
    return sf


def noise(shape, min, max, sigma=3):
    nf = np.random.uniform(0, 1, size=shape)
    kernel_size = 6*sigma - 1
    kernel = np.exp(-((np.arange(kernel_size)-(kernel_size/2))**2)/(sigma**2))
    # Here you would insert your actual kernel of any size
    nf = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode='same'), 0, nf)
    nf = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode='same'), 1, nf)
    return rescale(nf, min, max)


def register_data(image_data):
    # register image_data to server
    vti_data = ps.TrivialProducer(registrationName='vti_data')
    client = vti_data.GetClientSideObject()
    client.SetOutput(image_data)
    vti_data.UpdatePipeline(time=0.0)
    return vti_data


def persistence_diagram(data_proxy, sf_name='Scalars_'):
    ttk_pd = TTKPersistenceDiagram(
        registrationName='ttk_pd', Input=data_proxy, DebugLevel=0)
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
    ttk_persistence_simplification.VertexIdentifierField = [
        'POINTS', 'CriticalType']
    ttk_persistence_simplification.NumericalPerturbation = 1
    ps.UpdatePipeline(time=0.0, proxy=ttk_persistence_simplification)
    return ttk_persistence_simplification


def morse_complex(data_proxy):
    ttk_msc = TTKMorseSmaleComplex(
        registrationName='ttk_msc', Input=data_proxy, DebugLevel=0)
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


def extract_arcs(poly_data, image_size, origin):
    vtk_points = poly_data.GetPoints()
    n_cells = poly_data.GetNumberOfCells()
    arcs_image = np.zeros(image_size, dtype=np.float32)
    for cell_idx in range(n_cells):
        cell = poly_data.GetCell(cell_idx)
        n_pts = cell.GetNumberOfPoints()
        for vert_idx in range(n_pts):
            pid = cell.GetPointId(vert_idx)
            x, y, _ = np.round(np.array(vtk_points.GetPoint(
                pid), dtype=np.float32)-origin).astype(np.int32)
            arcs_image[x, y] = 1
    arcs_image.reshape([image_size[0], image_size[1], 1])
    return arcs_image


def save_simplified_image(image_data, sf_name, path):
    dims = image_data.GetDimensions()
    vtk_array = image_data.GetPointData().GetAbstractArray(sf_name)
    # assumes float array. change dtype if different. doesn't support bit array
    np_img = np.frombuffer(vtk_array, dtype=np.float32,
                           count=dims[0]*dims[1]*dims[2]).reshape(dims).transpose([1, 0, 2])
    np.save(path, np_img)


def prepare_output_directories(output_folder):
    # this function has side effect
    if output_folder[-1] != '/':
        output_folder += '/'
    images_folder = output_folder + 'originals/'
    simples_folder = output_folder + 'images/'
    arcs_folder = output_folder + 'arcs/'
    boundaries_folder = output_folder + 'boundaries/'
    dt_folder = output_folder + "distances/"
    check_dir(output_folder)
    check_dir(images_folder)
    check_dir(simples_folder)
    check_dir(arcs_folder)
    check_dir(boundaries_folder)
    check_dir(dt_folder)
    return output_folder, images_folder, simples_folder, arcs_folder, boundaries_folder, dt_folder


def distance_transform(image, threshold=0.5):
    w, h = image.shape
    p = np.zeros(image.shape)
    # 1p
    p[image <= threshold] = 2*(w+h)
    p[image > threshold] = 0
    for x in range(w):
        for y in range(h):
            # forward
            if x == 0 and y == 0:
                p[x, y] = p[x, y]
            elif x == 0:
                p[x, y] = np.min((p[x, y-1]+1, p[x, y]), axis=0)
            elif y == 0:
                p[x, y] = np.min((p[x-1, y]+1, p[x, y]), axis=0)
            else:
                p[x, y] = np.min((p[x-1, y]+1, p[x, y],
                                  p[x, y-1]+1), axis=0)
    for x in range(w):
        for y in range(h):
            inv_x = w-x-1
            inv_y = h-y-1
            if (inv_x == w-1) and (inv_y == h-1):
                p[inv_x, inv_y] = p[inv_x, inv_y]
            elif (inv_x == w-1):
                p[inv_x, inv_y] = np.min(
                    (p[inv_x, inv_y], p[inv_x, inv_y+1]+1), axis=0)
            elif (inv_y == h-1):
                p[inv_x, inv_y] = np.min(
                    (p[inv_x, inv_y], p[inv_x+1, inv_y]+1), axis=0)
            else:
                p[inv_x, inv_y] = np.min((p[inv_x, inv_y], p[inv_x+1, inv_y]+1,
                                          p[inv_x, inv_y+1]+1), axis=0)
    return p


def mc4image(image, idx, folders):
    sf_name = 'Scalars_'
    w, h, d = image.shape
    _, images_folder, simples_folder, arcs_folder, boundaries_folder, dt_folder = folders
    image = image.astype(np.float32)

    trans_image = image.transpose([2, 1, 0])
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(w, h, d)
    scalars = vtk.vtkFloatArray()
    scalars.SetNumberOfComponents(1)
    scalars.SetNumberOfTuples(w*h*d)
    scalars.SetName(sf_name)

    if not trans_image.flags.contiguous:
        np_slice = np.ascontiguousarray(trans_image)
    image_flat = np.ravel(np_slice)
    if not image_flat.flags.contiguous:
        image_flat = np.ascontiguousarray(image_flat)

    scalars.SetVoidArray(image_flat, image_flat.shape[0], 1)
    image_data.GetPointData().AddArray(scalars)
    image_data.GetPointData().SetActiveScalars(sf_name)

    image_proxy = register_data(image_data)
    ps.SaveData(f'{images_folder}original_{idx:06d}.vti', proxy=image_proxy,
                ChooseArraysToWrite=1, PointDataArrays=[sf_name], CellDataArrays=[])

    pd = persistence_diagram(image_proxy)
    threshold = persistence_threshold(pd, 0.04)
    simplified = simplification(image_proxy, threshold)
    ps.SaveData(f'{images_folder}simplified_{idx:06d}.vti', proxy=simplified,
                ChooseArraysToWrite=1, PointDataArrays=[sf_name], CellDataArrays=[])

    mc = morse_complex(image_proxy)
    # save arcs as vtp files
    poly_data = sm.Fetch(mc, idx=1)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(f'{arcs_folder}arcs_{idx:06d}.vtp')
    writer.SetInputData(poly_data)
    writer.Write()

    arcs_image = extract_arcs(
        poly_data, [w, h], np.array([0, 0, 0], dtype=np.float32))
    np.save(f'{boundaries_folder}boundary_{idx:06d}.npy', arcs_image)

    distance = np.copy(arcs_image)
    distance[distance == 1] = 2
    distance[distance == 0] = 1
    distance[distance == 2] = 0
    np.save(f'{dt_folder}distance_{idx:06d}.npy',
            ndimage.distance_transform_edt(distance))
    save_simplified_image(image_data, sf_name,
                          f'{simples_folder}simplified_{idx:06d}.npy')


def gen_data(image_size, group_size, folders, index):
    w, h = image_size
    params = []
    r = np.random.uniform(0, 1)
    if (r < 1/3):
        base_image = gen_image([w, h, 1], False)
        params.append([0])
    elif (r < 2/3):
        base_image = gen_image([w, h, 1], True)
        params.append([1])
    else:
        base_image = gen_blob([w, h, 1], 2, 512)
        params.append([2])
    # noise_image = base_image + np.random.uniform(0,0.05, size=base_image.shape)
    # mc4image(noise_image, index * group_size,folders )
    # params.append([param[0],0])
    for i in range(0, group_size):
        kernel_size = np.random.randint(3, 7)
        noise_image = base_image + \
            noise(base_image.shape, 0, 0.05, kernel_size)
        mc4image(noise_image, index * group_size + i, folders)
        params.append([params[0][0], kernel_size])
    return params


def run_range(image_size, img_range, group_size, output_dir):
    # output folders
    folders = prepare_output_directories(output_dir)
    output_folder, _, simplified_folder, _, boundaries_folder, dt_folder = folders
    simplified_index = open(f'{simplified_folder}index.txt', 'a')
    boundaries_index = open(f'{boundaries_folder}index.txt', 'a')
    distance_index = open(f'{dt_folder}index.txt', 'a')
    params_file = open(f'{output_folder}params.csv', 'a')

    # intialize local variables
    total = len(img_range)
    sum_iter_time = 0
    count = 0

    # main loop
    for i in img_range:
        t_start = datetime.now()

        params = gen_data(image_size, group_size, folders, i)

        for j in range(group_size):
            simplified_index.write(
                f'simplified_{i * group_size + j:06d}.npy\n')
            boundaries_index.write(f'boundary_{i * group_size + j:06d}.npy\n')
            distance_index.write(f'distance_{i * group_size + j:06d}.npy\n')
            params_file.write(f"{str(params[j])[1:-1]}\n")

        t_end = datetime.now()
        # log
        count += 1
        iter_time = (t_end - t_start).total_seconds()
        sum_iter_time += iter_time
        ave_iter_time = sum_iter_time/count
        sys.stdout.write(
            f'\rprocessed {count}/{total} data in {sum_iter_time:.03f} seconds at {ave_iter_time:.03f}s/item. est:{(total-count)*ave_iter_time:.03f}s')

    # close
    simplified_index.close()
    boundaries_index.close()
    distance_index.close()
    params_file.close()


def build_index(folder):
    if folder[-1] != '/':
        folder += '/'
    files = os.listdir(folder)
    with open(f'{folder}index.txt', 'w') as index_file:
        for file in files:
            index_file.write(f'{file}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='output directory')
    parser.add_argument('range', nargs=2, type=int)
    parser.add_argument('group_size', type=int)
    parser.add_argument('--image_size', nargs=2, type=int, default=[256, 256])
    opt = parser.parse_args()
    print(opt.range)
    print(opt.image_size)
    r = list(range(opt.range[0], opt.range[1]))
    run_range(opt.image_size, r, opt.group_size, opt.output)


if __name__ == '__main__':
    main()
