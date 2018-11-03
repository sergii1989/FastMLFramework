import sys
import logging

# Possible values here are DEBUG, INFO, WARN, ERROR and CRITICAL
LOG_LEVEL = logging.INFO  # logging level


def configure_logging():
    # TODO: to think whether to add or not logging to the file
    # logPath = r'c:\Kaggle\FastMLFramework\examples\classification\multiclass\iris'
    # fileName = 'demo.log'
    logging.basicConfig(format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
                        datefmt='%H:%M:%S',
                        level=LOG_LEVEL,
                        handlers=[
                            # logging.FileHandler(filename=os.path.join(logPath, fileName), mode='w'),
                            logging.StreamHandler(stream=sys.stdout)
                        ])
