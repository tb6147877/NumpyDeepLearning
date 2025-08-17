
class Layer():
    first_layer = False

    def forward(self, input, *args, **kwargs): # 动态参数
        raise NotImplementedError

    def backward(self, input, *args, **kwargs): # 动态参数
        raise NotImplementedError

    def connect_to(self,prev_layer):
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError

    @classmethod # 静态函数
    def from_json(cls,config):
        return cls(**config) #把参数解包进行构造

    @property
    def params(self):
        return []

    @property # 把一个方法伪装成属性。使用时，不需要加括号 ()，就像访问字段一样。
    def grads(self):
        return []

    @property
    def param_grads(self):
        return list(zip(self.params, self.grads)) # zip会把两个序列打包成对应的元组对

    def __str__(self): # tostring方法，会显示类名
        return self.__class__.__name__
