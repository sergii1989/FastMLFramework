from setuptools import setup

requirements = [
    "pip==18.1",
    "numpy==1.15.3",
    "matplotlib==2.2.3",
    "pandas==0.23.4",
    "scipy==1.1.0",
    "scikit-learn==0.20.0",
    "xgboost==0.80",
    "lightgbm==2.1.2",
    "shap==0.24.0",
    "graphviz==0.8.4",
    "pdpbox==0.2.0",
    "pyhocon==0.3.47",
    "cachetools==2.1.0",
    "future==0.16.0",
    "six==1.11.0",
    "bayesian-optimization==1.0.0",
    "seaborn",
    "luigi"
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
