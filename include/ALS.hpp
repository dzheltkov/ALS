#ifndef PBM2D_ALS
#define PBM2D_ALS

#include <cstdint>
#include <limits>
#include <algorithm>
#include <CXXBLAS.hpp>
#include <CXXLAPACK.hpp>
#include <iostream>
#include <chrono>

template<class DataType>
struct ALSParams
{
    typedef decltype(std::abs(DataType(0.0))) RealType;
    uint64_t block_size = 4096;
    uint64_t max_it = 200;
    RealType rel_tol = 1000.0 * std::numeric_limits<RealType>::epsilon();
    RealType abs_tol = 0.0;
};

    template <class Model, class DataType>
void ALS(Model &model, const DataType *rhs, const ALSParams<DataType> &params = ALSParams<DataType>())
{
    typedef decltype(std::abs(DataType(0.0))) RealType;
    const uint64_t D = model.dimensionality();
    const auto M = model.sizes();
    const uint64_t L = model.linear_size();

    const uint64_t K = model.length();

    const uint64_t max_M = *std::max_element(M.begin(), M.begin() + D);

    const uint64_t max_JN = max_M + L;

    const uint64_t block_size = std::min(params.block_size, K);

    const double rel_tol = params.rel_tol;
    const double abs_tol = params.abs_tol;
    uint64_t max_it = params.max_it;

    std::vector<DataType> J(block_size * max_JN);
    std::vector<DataType> B(max_JN);
    std::vector<DataType> H(max_JN * max_JN);
    std::vector<RealType> s(max_JN);

    auto nrm_squared = BLAS::nrm2(K, rhs, 1);
    nrm_squared *= nrm_squared;

    uint64_t d = 0;
    uint64_t i = 0;
    bool left_to_right = true;

    auto prev_err_squared = nrm_squared;
    std::chrono::duration<double> jac_gen_time;
    std::chrono::duration<double> herk_time;
    std::chrono::duration<double> other_time;
    while (true)
    {
        auto start_time = std::chrono::steady_clock::now();
        const uint64_t JN = M[d] + L;
        std::fill(B.begin(), B.begin() + JN, DataType(0.0));
        std::fill(H.begin(), H.begin() + JN * JN, DataType(0.0));
        auto end_time = std::chrono::steady_clock::now();

        other_time += std::chrono::duration<double>(end_time - start_time);

        for (uint64_t k = 0; k < K; k += block_size)
        {
            start_time = std::chrono::steady_clock::now();
            uint64_t k_last = std::min(k + block_size, K);
            model.JacobianPart(d, k, k_last, J.begin(), block_size);
            model.LinearPart(k, k_last, J.begin() + block_size * M[d], block_size);
            end_time = std::chrono::steady_clock::now();
            jac_gen_time += std::chrono::duration<double>(end_time - start_time);

            start_time = std::chrono::steady_clock::now();
            const uint64_t JM = k_last - k;

            BLAS::gemv('C', JM, JN, DataType(1.0), J.data(), block_size, rhs + k, 1, DataType(1.0), B.data(), 1);
            BLAS::herk('U', 'C', JN, JM, RealType(1.0), J.data(), block_size, RealType(1.0), H.data(), JN);
            end_time = std::chrono::steady_clock::now();

            herk_time += std::chrono::duration<double>(end_time - start_time);
        }

        start_time = std::chrono::steady_clock::now();
        int info = 0;

        RealType scond, amax;

        info = LAPACK::poequb(JN, H.data(), JN, s.data(), scond, amax);

        bool equed = LAPACK::laqhe('U', JN, H.data(), JN, s.data(), scond, amax);

        RealType alpha = 128 * std::numeric_limits<RealType>::epsilon();
        if (!equed)
        {
            alpha *= amax * scond;
        }
        else
        {
            for (uint64_t i = 0; i < JN; i++)
            {
                B[i] *= s[i];
            }
        }
        for (uint64_t i = 0; i < JN; i++)
        {
            H[i * (JN + 1)] += alpha;
        }


        info = LAPACK::potrf('U', JN, H.data(), JN);
        if (info != 0)
        {
            throw std::runtime_error("LAPACK::potrf failed");
        }
        BLAS::trsv('U', 'C', 'N', JN, H.data(), JN, B.data(), 1);
        auto err_squared = BLAS::nrm2(JN, B.data(), 1);
        err_squared = nrm_squared - err_squared * err_squared;
        BLAS::trsv('U', 'N', 'N', JN, H.data(), JN, B.data(), 1);
        if (equed)
        {
            for (uint64_t i = 0; i < JN; i++)
            {
                B[i] *= s[i];
            }
        }
        model.update(d, B.begin());
        model.update_linear(B.begin() + M[d]);
        end_time = std::chrono::steady_clock::now();

        other_time += std::chrono::duration<double>(end_time - start_time);
        if (left_to_right)
        {
            d++;
            if (d == D - 1)
            {
                left_to_right = false;
            }
        }
        else
        {
            d--;
            if (d == 0)
            {
                left_to_right = true;
                i++;
                std::cout << i << ' ' << 10 * (std::log10(err_squared) - std::log10(nrm_squared)) << ' ' << (prev_err_squared - err_squared) / nrm_squared 
                          << ' ' << jac_gen_time.count() << ' ' << herk_time.count() << ' ' << other_time.count() << std::endl;

                if ((prev_err_squared - err_squared) <= std::min(abs_tol, rel_tol * nrm_squared) || i == max_it)
                {
                    break;
                }
                prev_err_squared = err_squared;
            }
        }
    }
    std::cout << 20 * std::log10(model.residual_norm()) - 10  * std::log10(nrm_squared) << std::endl;
}
#endif
