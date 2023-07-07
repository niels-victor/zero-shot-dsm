# Add installed cuda runtime to path for bitsandbytes 
import os
import nvidia


def setup_cuda():
    cuda_install_dir = '/'.join(nvidia.__file__.split('/')[:-1]) + '/cuda_runtime/lib/'
    os.environ['LD_LIBRARY_PATH'] =  cuda_install_dir
