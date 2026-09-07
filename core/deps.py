from functools import cache, lru_cache

from services.file_import_service import ImportFileService

@cache    #  将创建业务对象进行缓存区，可以重复利用。 缓存长期有效。
#@lru_cache #  缓存满了，根据LRU(最近最少使用)算法清理缓存。
def get_import_file_service():
    return ImportFileService()