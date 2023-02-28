import os

def check_dir(path):
    if not (os.path.exists(path) and os.path.isdir(path)):
        os.makedirs(path)

def create_symlink(src,dst):
    os.symlink(src, dst)

def read_params(param_file):
    lines = []
    with open(param_file, 'r') as infile:
        for line in infile:
            lines.append(line.strip())
    return lines

def read_index_file(index_file):
    file_names = []
    try:
        with open(index_file, 'r') as infile:
            for line in infile:
                file_name = line.strip()
                if (file_name):
                    file_names.append(file_name)
    except:
        pass
    return file_names

def link_folder(src, dst, prefix):
    dst_index_file = dst + 'index.txt'
    file_count = len(read_index_file(dst_index_file))
    file_names = read_index_file(src + 'index.txt')
    count = 0;
    with open(dst_index_file, 'a') as index_file:
        for file_name in file_names:
            target_name = f'{prefix}_{count + file_count:05d}.npy'
            index_file.write(f'{target_name}\n')
            create_symlink(f'{src}{file_name}', f'{dst}{target_name}')
            count += 1

def merge_params(src, dst):
    with open(f'{dst}params.csv','a') as outfile:
        with open(f'{src}params.csv','r') as infile:
            for line in infile:
                outfile.write(line)

def merge_data_into(src, dst):
    link_folder(src + 'boundaries/', dst + 'boundaries/', 'boundary')
    link_folder(src + 'distances/', dst + 'distances/', 'distance')
    link_folder(src + 'images/', dst + 'images/', 'simplified')
    
    merge_params(src, dst)

def merge(input_folders, output_folder):
    check_dir(output_folder)
    out_boundary_folder = output_folder + 'boundaries/'
    check_dir(out_boundary_folder)

    out_image_folder = output_folder + "images/"
    check_dir(out_image_folder)

    out_distance_folder = output_folder + "distances/"
    check_dir(out_distance_folder)

    for input_folder in input_folders:
        merge_data_into(input_folder, output_folder)


def main():
    input_folders = [f'CREMI_AB_64_64_train_{i}/' for i in range(20)]
    input_folders = list(map(lambda x:os.path.abspath(x)+'/', input_folders))
    output_folder = 'CREMI_AB_64_64_train/'
    merge(input_folders, output_folder)

    input_folders = [f'CREMI_AB_64_64_test_{i}/' for i in range(20)]
    input_folders = list(map(lambda x:os.path.abspath(x)+'/', input_folders))
    output_folder = 'CREMI_AB_64_64_test/'
    merge(input_folders, output_folder)


if __name__ == '__main__':
    main()
