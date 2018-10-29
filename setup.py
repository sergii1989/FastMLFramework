import pip
from setuptools import setup
from setuptools.command.install import install

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
    "seaborn",
    "luigi",
    "bayesian-optimization"
]


class OverrideInstallCommand(install):
    def run(self):

        # Install all requirements
        failed = []
        for req in requirements:
            if pip.main(["install", req]) == 1:
                failed.append(req)

        if len(failed) > 0:
            print("")
            print("Error installing the following packages:")
            print(str(failed))
            print("Please install them manually")
            print("")
            raise OSError("Aborting")

        # install FastMLFramework
        install.run(self)


with open('README.md') as readme_file:
    readme = readme_file.read()


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
    long_description=readme,
    cmdclass={'install': OverrideInstallCommand},
    install_requires=requirements,
    zip_safe=False,
    keywords='auto-ml ML pipeline luigi features selection stacking blending hyper-parameters optimization',
)
