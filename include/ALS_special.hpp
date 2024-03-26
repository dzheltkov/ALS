#ifndef PBM2D_ALS
#define PBM2D_ALS

#include <cstdint>
#include <limits>
#include <algorithm>
#include <CXXBLAS.hpp>
#include <CXXLAPACK.hpp>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <iomanip>
#include <functional>

template<class DataType>
struct ALSParams
{
    typedef decltype(std::abs(DataType(0.0))) RealType;
    uint64_t block_size = 128;
    uint64_t max_it = 200;
    RealType rel_tol = 1024 * std::numeric_limits<RealType>::epsilon();
    RealType abs_tol = 0.0;
    //std::function<RealType (int64_t, const DataType *, const DataType *)> additional_metric = nullptr; 
    std::function<std::complex<double> (int64_t, const std::complex<double> *, const std::complex<double> *)> additional_metric = nullptr;
};

    template <class Model, class DataType>
void ALS(Model &model, const ALSParams<DataType> &params = ALSParams<DataType>())
{
    typedef decltype(std::abs(DataType(0.0))) RealType;

    const DataType *rhs;
    const uint64_t D = model.dimensionality();
    const auto M = model.sizes();
    const uint64_t L = model.linear_size();

    const uint64_t K = model.length_for_ALS();

    const uint64_t max_M = *std::max_element(M.begin(), M.begin() + D);

    const uint64_t max_JN = max_M + L;

    const uint64_t block_size = std::min(params.block_size, K);

    const double rel_tol = params.rel_tol;
    const double abs_tol = params.abs_tol;
    uint64_t max_it = params.max_it;

    std::vector<DataType> R(K);

    uint64_t d = 0;
    uint64_t i = 0;
    bool left_to_right = true;

    rhs = model.rhs(d);
    auto nrm = BLAS::nrm2(K, rhs, 1);

    auto prev_err = nrm;
    std::chrono::duration<double> jac_gen_time;
    std::chrono::duration<double> other_time;
    std::vector<std::vector<DataType> > J(omp_get_max_threads(), std::vector<DataType>(block_size * max_JN));
    std::vector<std::vector<DataType> > H(omp_get_max_threads(), std::vector<DataType>(max_JN * max_JN));
    std::vector<std::vector<DataType> > B(omp_get_max_threads(), std::vector<DataType>(max_JN));
    while (true)
    {
        rhs = model.rhs(d);
        //std::cout <<"start d:" << d << " " << rhs[0] << " " << rhs[1] << " " << rhs[2] << std::endl;
        auto start_time = std::chrono::steady_clock::now();
        const uint64_t JN = M[d] + L;
        auto end_time = std::chrono::steady_clock::now();

        other_time += std::chrono::duration<double>(end_time - start_time);

        start_time = std::chrono::steady_clock::now();
#pragma omp parallel
        {
            int t = omp_get_thread_num();
            std::fill(B[t].begin(), B[t].begin() + JN, DataType(0.0));
            std::fill(H[t].begin(), H[t].begin() + JN * JN, DataType(0.0));
#pragma omp for
            for (uint64_t k = 0; k < K; k += block_size)
            {
                uint64_t k_last = std::min(k + block_size, K);
                model.JacobianPart(d, k, k_last, J[t].begin(), block_size);
                model.LinearPart(k, k_last, J[t].begin() + block_size * M[d], block_size);

                const uint64_t JM = k_last - k;

                BLAS::gemv('N', JM, M[d], DataType(1.0), J[t].data(), block_size, model.mode(d), 1, DataType(0.0), R.data() + k, 1);
                BLAS::gemv('N', JM, L, DataType(1.0), J[t].data() + block_size * M[d], block_size, model.linear(), 1, DataType(1.0), R.data() + k, 1);
                BLAS::gemv('C', JM, JN, DataType(1.0), J[t].data(), block_size, rhs + k, 1, DataType(1.0), B[t].data(), 1);
                BLAS::herk('U', 'C', JN, JM, RealType(1.0), J[t].data(), block_size, RealType(1.0), H[t].data(), JN);
            }
#pragma omp for nowait
            for (uint64_t j = 0; j < JN * JN; j++)
            {
                for (int64_t l = 1; l < omp_get_num_threads(); l++)
                {
                    H[0][j] += H[l][j];
                }
            }
#pragma omp for nowait
            for (uint64_t j = 0; j < JN; j++)
            {
                for (int64_t l = 1; l < omp_get_num_threads(); l++)
                {
                    B[0][j] += B[l][j];
                }
            }
        }
        end_time = std::chrono::steady_clock::now();
        jac_gen_time += std::chrono::duration<double>(end_time - start_time);

        start_time = std::chrono::steady_clock::now();
        int info = 0;

        RealType alpha = 1024 * std::numeric_limits<RealType>::epsilon();
        for (uint64_t i = 0; i < JN; i++)
        {
            H[0][i * (JN + 1)] *= 1 + alpha;
        }

        info = LAPACK::potrf('U', JN, H[0].data(), JN);
        if (info != 0)
        {
            throw std::runtime_error("LAPACK::potrf failed");
        }
        LAPACK::potrs('U', JN, 1, H[0].data(), JN, B[0].data(), JN);
        model.update(d, B[0].begin());
        model.update_linear(B[0].begin() + M[d]);
        end_time = std::chrono::steady_clock::now();

        other_time += std::chrono::duration<double>(end_time - start_time);
        //std::cout << std::setprecision(10) << " " << d << " residual_error: " << 20 * (std::log10(model.residual_norm()) - std::log10(nrm)) << std::endl;

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
                start_time = std::chrono::steady_clock::now();
                left_to_right = true;
                i++;
                std::complex<double> add_err = 0;
                if (params.additional_metric)
                {
                    add_err = params.additional_metric(model.length(), reinterpret_cast<const std::complex<double>*>(rhs), reinterpret_cast<std::complex<double>*>(R.data()));
                }
                //std::cout <<"end d:" << d << " " << rhs[0] << " " << rhs[1] << " " << rhs[2] << std::endl;
                BLAS::axpy(K, DataType(-1.0), rhs, 1, R.data(), 1);
                auto err = BLAS::nrm2(K, R.data(), 1);
                end_time = std::chrono::steady_clock::now();

                other_time += std::chrono::duration<double>(end_time - start_time);
                std::cout << std::setprecision(7) << i << ' ' << 20 * (std::log10(err) - std::log10(nrm)) << ' ';
                if (params.additional_metric)
                {
                    std::cout << add_err.real() << ' ';
                }
                std::cout << (prev_err - err) / nrm
                          << ' ' << jac_gen_time.count() << ' ' << other_time.count() << std::endl;

                std::fflush(stdout);
                model.logging_error(i, 20 * (std::log10(err) - std::log10(nrm)),
                    (prev_err - err) / nrm, jac_gen_time.count(),
                    other_time.count(), add_err);

                if (i % 25 == 0) {
                    model.logging_coef(i);
                }

                if ((prev_err - err) <= std::max(abs_tol, rel_tol * nrm) || i == max_it)
                {
                    break;
                }
                prev_err = err;
            }
        }
    }
}
#endif
