import sys
import logging


def configure_logging():
    # logPath = r'c:\Kaggle\FastMLFramework\examples\classification\multiclass\iris'
    # fileName = 'demo.log'
    logging.basicConfig(format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO,
                        handlers=[
                            # logging.FileHandler(filename=os.path.join(logPath, fileName), mode='w'),
                            logging.StreamHandler(stream=sys.stdout)
                        ])