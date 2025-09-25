import os
DATAROOT=os.getenv('DATAROOT', 'data/test')
MAX_INPUT_LEN=int(os.getenv('MAX_INPUT_LEN', '120000'))
MAX_OUTPUT_LEN=int(os.getenv('MAX_OUTPUT_LEN', '10000'))
URL = f"http://{os.getenv('SERVE_HOST', '127.0.0.1')}:{os.getenv('SERVE_PORT', '8000')}/v1"
API_KEY = "123-abc"
RECURRENT_MAX_CONTEXT_LEN=int(os.getenv('RECURRENT_MAX_CONTEXT_LEN', '120000'))
RECURRENT_CHUNK_SIZE=int(os.getenv('RECURRENT_CHUNK_SIZE', '5000'))
RECURRENT_MAX_NEW=int(os.getenv('RECURRENT_MAX_NEW', '1024'))
from pprint import pprint
pprint({
    k: v
    for k, v in globals().items()
    if k in os.environ
})