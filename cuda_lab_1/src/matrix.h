#pragma once

#include <cstring>
#include <cmath>
#include <algorithm>

namespace
{
    bool fuzzy_comp(float a, float b, float max_err = 1e-3)
    {
        if(abs(a - b) < max_err)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

struct SquareMatrix
{
    explicit SquareMatrix(size_t sz_ = 0, float fill_val = 0.) :
        sz{sz_}
    {
        m = create_squre_matrix(sz);
        fill(fill_val);
    }

    SquareMatrix(const SquareMatrix &other) :
        sz{other.sz}
    {
         m = create_squre_matrix(sz);
         std::memcpy(m, other.m, sz*sz*sizeof(float));
    }

    ~SquareMatrix()
    {
        delete[] m;
    }

    bool operator==(const SquareMatrix &rhs) const
    {
        if(sz != rhs.sz)
        {   
            return false;
        }

        for(size_t i = 0; i < sz; ++i)
        {
            for(size_t j = 0; j < sz; ++j)
            {
                if(!fuzzy_comp(m[i*sz + j], rhs.m[i*sz + j], 1e-2))
                {
                    // std::cout << i << ' ' << j << ' ' << m[i*sz + j] << " != " << rhs.m[i*sz + j] << '\n';
                    return false;
                }
            }
        }

        return true;
    }

    // well...we can't overload operator [][]
    float get_val(int i, int j) const
    {
        if(i < 0 || j < 0 || i >= sz || j >= sz)
        {
            return 0;
        }
        else
        {
            return m[i*sz + j];
        }
    }

    float *operator[](size_t i)
    {
        return &m[i*sz];
    }

    const float *operator[](size_t i) const
    {
        return &m[i*sz];
    }

    float *data()
    {
        return m;
    }

    const float *data() const 
    {
        return m;
    }

    size_t size() const
    {
        return sz;
    }

    const SquareMatrix &operator=(SquareMatrix && rhs)
    {
        sz = rhs.sz;
        m = rhs.m;

        rhs.m = nullptr;
        rhs.sz = 0;

        return *this;
    }

    void fill(const float val)
    {
        std::fill_n(m, sz*sz, val);
    }

private:
    float *create_squre_matrix(size_t size)
    {
        return new float[size*size];
    }

private:
    float *m; 
    size_t sz;
};