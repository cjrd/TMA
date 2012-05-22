import pdb
from django.core.files.uploadhandler import FileUploadHandler, StopUpload
from src import settings


class FSUploadHandler(FileUploadHandler):
    """
    This uploadhandler limits the size of uploaded files as determined by settings.MAX_UPLOAD_SIZE
    """

    def __init__(self, request=None):
        super(FSUploadHandler, self).__init__(request)

    def handle_raw_input(self, input_data, META, content_length, boundary, encoding=None):
        # TODO what if data does not specify content_length?
        if content_length > settings.MAX_UPLOAD_SIZE:
            raise StopUpload(connection_reset=True)

    def receive_data_chunk(self, raw_data, start):
        # TODO add more security checks
        return raw_data

    def file_complete(self, file_size):
        return None