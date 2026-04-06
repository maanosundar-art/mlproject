from setuptools import find_packages, setup
from  typing import List

def get_requirements(file_path:str) -> List[str]:
    with open(file_path) as file:
        req= file.readlines()
        requirements = [r.replace('\n','') for r in req]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements


setup(name='mlproject1',
      version='0.0.1',
      author='maano',
      author_email='maanosundar@gmail.com',
      packages=find_packages(),
      install_requirements= get_requirements('requirements.txt'))