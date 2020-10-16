#
from apps.dcl.mdl.m_bmy import MBmy
from apps.dcl.mdl.m_bm import MBm

class CWxs(object):
    def _init__(self):
        self.refl = ''

    @staticmethod
    def get_bmy_id_bm_vo_dict():
        '''
        获取bmy_id（年款头输出）与车型值对象的字典
        '''
        bmy_id_bm_vo_dict = {}
        bmy_id_model_ids = MBmy.get_bmy_id_model_ids()
        for bimi in bmy_id_model_ids:
            bmy_id = int(bimi['bmy_id'])
            model_id = int(bimi['model_id'])
            model_vo = MBm.get_model_vo_by_id(model_id)
            bm_vo = {
                'model_id': model_id,
                'model_name': model_vo['model_name'],
                'model_code': model_vo['model_code'],
                'source_type': model_vo['source_type']
            }
            bmy_id_bm_vo_dict[bmy_id] = bm_vo
        return bmy_id_bm_vo_dict