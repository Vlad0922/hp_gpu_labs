#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <cstring>
#include <assert.h>

const char *INPUT_FNAME  = "input.txt";
const char *OUTPUT_FNAME = "output.txt";

struct SquareMatrix
{
    explicit SquareMatrix(size_t sz_ = 0) :
        sz{sz_}
    {
        m = create_squre_matrix(sz);
    }

    SquareMatrix(const SquareMatrix &other) :
        sz{other.sz}
    {
         m = create_squre_matrix(sz);

         for(size_t i = 0; i < sz; ++i)
         {
             std::memcpy(m[i], other.m[i], sz*sizeof(double));
         }
    }

    ~SquareMatrix()
    {
        for(size_t i = 0; i < sz; ++i)
        {
            delete[] m[i];
        }

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
                if(m[i][j] != rhs.m[i][j])
                {
                    return false;
                }
            }
        }

        return true;
    }

    // well...we can't overload operator [][]
    double get_val(int i, int j) const
    {
        if(i < 0 || j < 0 || i >= sz || j >= sz)
        {
            return 0;
        }
        else
        {
            return m[i][j];
        }
    }

    double *operator[](size_t i)
    {
        return m[i];
    }

    const double *operator[](size_t i) const
    {
        return m[i];
    }

    double **data()
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

private:
    double **create_squre_matrix(size_t size)
    {
        double **arr = new double*[size];

        for(size_t i = 0; i < size; ++i)
        {
            arr[i] = new double[size];
        }

        return arr;
    }
private:
    double **m;
    size_t sz;
};

void read_from_fstream(std::ifstream &input, double **arr, const size_t size)
{
    double val;
    for(size_t i = 0; i < size; ++i)
    {
        for(size_t j = 0; j < size; ++j)
        {
            input >> val;
            arr[i][j] = val;
        }
    }
}

std::tuple<SquareMatrix, SquareMatrix> read_data()
{
    std::ifstream input_file(INPUT_FNAME);

    if(!input_file.is_open())
    {
        std::cerr << "Cannot open input.txt!\n";
        exit(1);
    }

    size_t N, M;

    input_file >> N >> M;

    SquareMatrix A(N);
    SquareMatrix B(M);

    read_from_fstream(input_file, A.data(), N);
    read_from_fstream(input_file, B.data(), M);

    return std::make_tuple(A, B);
}

void print_matrix(const SquareMatrix &m)
{
    for(size_t i = 0; i < m.size(); ++i)
    {
        for(size_t j = 0; j < m.size(); ++j)
        {
            std::cout << m[i][j] << ' ';
        }
        std::cout << '\n';
    }
}


double calc_cell_convolve(const SquareMatrix &A, const SquareMatrix &B, int i, int j)
{
    double val = 0;
    int HM = (B.size() - 1)/2;

    for(int k = -HM; k <= HM; ++k)
    {
        for(int l = -HM; l <= HM; ++l)
        {
            val += A.get_val(i+k, j+l)*B.get_val(k+HM, l+HM);
        }
    }

    return val;
}


// simple function to check the result
SquareMatrix convolve(const SquareMatrix &A, const SquareMatrix &B)
{
    SquareMatrix C(A.size());

    for(int i = 0; i < A.size(); ++i)
    {
        for(int j = 0; j < A.size(); ++j)
        {
            C[i][j] = calc_cell_convolve(A, B, i, j);
        }
    }

    return C;
}

int main()
{
    SquareMatrix A, B;
    std::tie(A, B) = read_data();

    SquareMatrix C = convolve(A, B);
    std::cout << "Ok!\n";

    print_matrix(C);

    return 0;
}
