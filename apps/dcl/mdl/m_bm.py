#

from os import stat
import pymongo
from apps.dcl.mdl.m_mongodb import MMongoDb

class MBm(object):
    def __init__(self):
        self.name = ''

    @staticmethod
    def get_model_vo_by_id(model_id):
        query_cond = {'model_id': model_id}
        fields = {'model_name': 1, 'model_code': 1, 'source_type': 1}
        return MMongoDb.db['t_model'].find_one(query_cond, fields)