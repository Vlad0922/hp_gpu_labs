#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>

#include <assert.h>

#include "matrix.h"

#include "convolve.cu"

const char *INPUT_FNAME  = "input.txt";
const char *OUTPUT_FNAME = "output.txt";

void read_from_fstream(std::ifstream &input, float *arr, const size_t size)
{
    float val;
    for(size_t i = 0; i < size; ++i)
    {
        for(size_t j = 0; j < size; ++j)
        {
            input >> val;
            arr[i*size + j] = val;
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


float calc_cell_convolve(const SquareMatrix &A, const SquareMatrix &B, int i, int j)
{
    float val = 0;
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

    SquareMatrix C = convolve_with_cuda(A, B);
    std::cout << "Ok!\n";

    print_matrix(C);

    return 0;
}
