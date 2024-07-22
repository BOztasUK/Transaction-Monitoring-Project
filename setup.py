from setuptools import find_packages, setup
from typing import List

H = '-e .'
def get_requirenments(file_path:str) ->List[str]:
    '''
    this function will return the list of requirenments
    '''
    requirenments = []
    with open(file_path) as file_obj:
        requirenments=file_obj.readlines()
        [req.replace("\n","") for req in requirenments]

        if H in requirenments:
            requirenments.remove(H)
    
    return requirenments

setup(
name='TM-Project',
version = "0.0.1",
author = 'Berkan',
author_email='berkanoztasuk@gmail.com',
packages=find_packages(),
install_requires = get_requirenments('requirements.txt')
)