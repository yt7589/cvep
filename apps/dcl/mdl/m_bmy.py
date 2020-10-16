# 
from os import stat
import pymongo
from apps.dcl.mdl.m_mongodb import MMongoDb

class MBmy(object):
    def __init__(self):
        self.name = ''

    @staticmethod
    def get_bmy_id_model_ids():
        '''
        获取bmy_id对应model_id的对应关系
        '''
        query_cond = {}
        fields = {'bmy_id': 1, 'model_id': 1}
        return MMongoDb.convert_recs(MMongoDb.db['t_bmy']\
                    .find(query_cond, fields))