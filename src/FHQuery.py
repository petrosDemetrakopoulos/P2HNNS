from copy import deepcopy
from .Query import Query

class FHQuery(Query):
    def __init__(self, query, data, top, limit, l, dist):
        super().__init__(query, data, top, limit, dist)
        self.l = l

    def copy(self, dist):
        return FHQuery(deepcopy(self.query), deepcopy(self.data), self.top, self.limit, self.l, dist)
