from setuptools import setup

######################################################################################################
################ You May Remove All the Comments Once You Finish Modifying the Script ################
######################################################################################################

setup(

    name = 'ecolaf', 
    
    version = '0.1.0',

    description = 'A toolkit python package to build an evidential multimodal neural network for semantic segmentation or classification.',
    
    py_modules = ["ECOLAF"],

    package_dir = {'':'src'},
    
    author = 'Lucas Deregnaucourt',
    author_email = 'lucas.deregnaucourt@insa-rouen.fr',
    
    long_description = open('README.md').read() + '\n\n' + open('CHANGELOG.md').read(),
    long_description_content_type = "text/markdown",
    
    url='https://github.com/jinhangjiang/morethansentiments',
    
    '''
    Leave it as deafult.
    '''
    include_package_data=True,
    
    '''
    This is not a enssential part. It will not affect your package uploading process. 
    But it may affect the discoverability of your package on pypi.org
    Also, it serves as a meta description written by authors for users.
    Here is a full list of what you can put here:
    
    https://pypi.org/classifiers/
    
    '''

    '''
    classifiers  = [
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: BSD License",
        'Intended Audience :: Developers',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Text Processing',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: OS Independent',
    ],
    '''
    
    '''
    This part specifies all the dependencies of your package. 
    "~=" means the users will need a minimum version number of the dependecies to run the package.
    If you specify all the dependencies here, you do not need to write a requirements.txt separately like many others do.
    '''
    install_requires = [

        'torch ~= 2.0.0'

    ],
    
    
    
    '''
    The keywords of your package. It will help users to find your package on pypi.org
    '''
    keywords = ['Dempster-Shafer Theory', 'Deep Learning', 'Multimodal', 'Semantic Segmentation', 'Classification'],
    
)
