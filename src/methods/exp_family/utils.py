import os
import sys
sys.path.append(os.getcwd())

def list_add(*param_lists):
    return [
        sum(params)
        for
        params
        in
        zip(*param_lists)
    ]

def list_sub(a, b):
    return [
        ai - bi
        for 
        ai, bi
        in
        zip(a, b)
    ]

def list_mul(a, b):
    return [
        ai * bi
        for 
        ai, bi
        in
        zip(a, b)
    ]

def list_const_mul(const, a):
    return [
        const * ai
        for
        ai 
        in 
        a
    ] 

def list_const_div(const, a):
    return [
        ai / const
        for
        ai 
        in 
        a
    ]

def list_to_numpy(a):
    return [
        ai.to_numpy()
        for ai
        in a
    ]

def list_to_torch(a, device):
    return [
        ai.to_torch(device)
        for ai
        in a
    ]

if __name__ == '__main__':
    from src.methods.exp_family.diagonal_gaussian import DiagGaussianNatParams
    print(list_add(
        [1,2,3],
        [1,2,3],
        [4,5,6]
    ))

    print(list_add(
        [
            DiagGaussianNatParams(1.,2.),
            DiagGaussianNatParams(1.,2.)
        ],
        [
            DiagGaussianNatParams(1.,2.),
            DiagGaussianNatParams(1.,2.)
        ],
    ))