import run_model


def linear():
    print('从零开始实现线性回归模型：')
    run_model.linear_scratch()
    print('简洁实现线性回归模型：')
    run_model.linear_concise()


def softmax():
    print('从零开始实现softmax分类模型：')
    run_model.softmax_scratch()
    print('简洁实现softmax分类模型：')
    run_model.softmax_concise()


if __name__ == "__main__":
    linear()
    softmax()