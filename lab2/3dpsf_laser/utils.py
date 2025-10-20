import glob

def list_image_files(images_dir):
    ext = ['png', 'jpg', 'JPG', 'gif'] 
    files = []
    [files.extend(glob.glob(images_dir + '/*.' + e)) for e in ext]
    return files