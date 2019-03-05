from setuptools import setup

requirements = [
    "pip==19.0.3",
    "numpy==1.16.2",
    "matplotlib==3.0.3",
    "pandas==0.24.1",
    "scipy==1.2.1",
    "scikit-learn==0.20.3",
    "xgboost==0.82",
    "lightgbm==2.2.3",
    "shap==0.28.5",
    "graphviz==0.10.1",
    "pdpbox==0.2.0",
    "pyhocon==0.3.51",
    "cachetools==3.1.0",
    "future==0.17.1",
    "six==1.12.0",
    "bayesian-optimization==1.0.1",
    "seaborn==0.9.0",
    "luigi==2.8.3"
]


setup(
    name='FastMLFramework',
    version='0.1.1',
    packages=['', 'data_vis', 'modeling', 'ensembling', 'ensembling.blending', 'ensembling.stacking', 'generic_tools',
              'data_processing', 'solution_pipeline'],
    url='https://github.com/sergii1989/FastMLFramework',
    license='BSD - 3',
    author='Sergii Lutsanych',
    author_email='sergii.lutsanych@gmail.com',
    description='FastML allows construction of ML end-to-end solutions in systematic, organized, fault-tolerant, and '
                'semi-automatic manner leveraging power of luigi pipelines',
    install_requires=requirements,
    keywords='auto-ml ML pipeline luigi features selection stacking blending hyper-parameters optimization',
)
