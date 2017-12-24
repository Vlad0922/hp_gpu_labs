#include <iostream>
#include <fstream>
#include <random>
#include <chrono>

#include "prefix.cu"

std::vector<float> read_data(const char *fname)
{
    std::ifstream input(fname);

    size_t sz;
    input >> sz;

    std::vector<float> data(sz);

    for(size_t i = 0; i < sz; ++i)
    {
        float val;
        input >> val;

        data[i] = val;
    }

    return data;
}

std::vector<float> generate_random_data(size_t sz)
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0., 5.);

    std::vector<float> data(sz);

    for(size_t i = 0; i < sz; ++i)
    {
        data[i] = distribution(generator);
    }

    return data;
}

using test_val_t = std::pair<std::vector<float>, const char *>;
void test_prefix()
{
    std::vector<test_val_t> tests = {{generate_random_data(32), "1st test, 32"},
                                     {generate_random_data(1024), "2nd test, 1024"},
                                     {std::vector<float>(1024, 0.), "3rd test, 1024 zeros"},
                                     {generate_random_data(1024*16), "4th test, 1024*16"}};

    for(auto &t : tests)
    {
        std::vector<float> cuda_res = calc_prefix_cuda(t.first);
        std::vector<float> correct = calc_prefix_cpu(t.first);

        if(!std::equal(cuda_res.begin(), cuda_res.end(), correct.begin(), [](float x, float y){return abs(x - y) < 1e-3;}))
        {
            std::cout << t.second << " failed!\n";
        }
        else
        {
            std::cout << t.second << " ok!\n";
        }
    }
}

void test_time(const std::string &fname = "cpu_vs_gpu.csv")
{
    std::cout << "testing time...\n";

    size_t start = 128;
    size_t step = 128;
    size_t end = 1 << 20;

    std::ofstream out(fname);

    if(!out.is_open())
    {
        std::cerr << "cannot open " << fname << "!\n";
        return;
    }

    out << "size;cpu;gpu\n";

    size_t steps_count = (end - start) / step;
    size_t curr_step = 0;
    double progress = 0.;

    int barWidth = 50;

    std::cout << "[";

    for(size_t sz = start; sz <= end; sz += step)
    {
        std::vector<float> data = generate_random_data(sz);

        auto start = std::chrono::steady_clock::now();
        calc_prefix_cpu(data);
        auto end = std::chrono::steady_clock::now();

        double cpu_time = std::chrono::duration<double, std::milli> (end - start).count();

        start = std::chrono::steady_clock::now();
        calc_prefix_cuda(data);
        end = std::chrono::steady_clock::now();

        double gpu_time = std::chrono::duration<double, std::milli> (end - start).count();

        out << sz << ';' << cpu_time << ';' << gpu_time << '\n';

        curr_step += 1;

        // copied from https://stackoverflow.com/questions/14539867/how-to-display-a-progress-indicator-in-pure-c-c-cout-printf
        progress = 1.*curr_step/steps_count;
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();
    }

    std::cout << "done!\n";
}

int main(int argc, char **argv)
{
    test_prefix();
    test_time();
}