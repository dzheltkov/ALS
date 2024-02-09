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
#include <functional>

template<class DataType>
struct ALSParams
{
    typedef decltype(std::abs(DataType(0.0))) RealType;
    uint64_t block_size = 64;
    uint64_t max_it = 100;
    RealType rel_tol = 1024 * std::numeric_limits<RealType>::epsilon();
    RealType abs_tol = 0.0;
    std::function<RealType (int64_t, const DataType *, const DataType *)> additional_metric = nullptr;
};

    template <class Model, class DataType>
void ALS(Model &model, const DataType *rhs, const ALSParams<DataType> &params = ALSParams<DataType>())
{
    typedef decltype(std::abs(DataType(0.0))) RealType;
    const uint64_t D = model.dimensionality();
    const auto M = model.sizes_for_solution();  // !!!!! sizes for real solution
    const uint64_t L = model.linear_size();   // here size is still for complex solution -- better change?

    const uint64_t K = model.length();

    const uint64_t max_M = *std::max_element(M.begin(), M.begin() + D);

    const uint64_t max_JN = max_M + 2*L;  // 2L because linear coeffs are also (real + imag) in this version 

    const uint64_t block_size = std::min(params.block_size, K);

    const double rel_tol = params.rel_tol;
    const double abs_tol = params.abs_tol;
    uint64_t max_it = params.max_it;

    // real and complex representation of the same object:
    std::vector<RealType> Re(2*K);
    std::vector<DataType> R(K);

    auto nrm = BLAS::nrm2(K, rhs, 1);

    uint64_t d = 0;
    uint64_t i = 0;
    bool left_to_right = true;

    auto prev_err = nrm;
    std::chrono::duration<double> jac_gen_time;
    std::chrono::duration<double> other_time;
    // std::vector<std::vector<DataType> > J(omp_get_max_threads(), std::vector<DataType>(block_size * max_JN));
    // std::vector<std::vector<DataType> > H(omp_get_max_threads(), std::vector<DataType>(max_JN * max_JN));
    // std::vector<std::vector<DataType> > B(omp_get_max_threads(), std::vector<DataType>(max_JN));
    std::vector<std::vector<RealType> > J(omp_get_max_threads(), std::vector<RealType>(2*block_size * max_JN));  // 2*block_size because the number of equations is (x2) - for real and imag parts
    std::vector<std::vector<RealType> > H(omp_get_max_threads(), std::vector<RealType>(max_JN * max_JN));
    std::vector<std::vector<RealType> > B(omp_get_max_threads(), std::vector<RealType>(max_JN));

    std::vector<RealType> rhs_real(2*K);  // real represetnation of the rhs
    for (uint64_t k = 0; k < K; k ++) {
        rhs_real[2*k] = rhs[k].real();
        rhs_real[2*k + 1] = rhs[k].imag();
    }

    while (true)
    {
        auto start_time = std::chrono::steady_clock::now();
        const uint64_t JN = M[d] + 2*L;   // M = {2 * 2 * (_M + 1) * _R, 2 * 2 * (_N + 1) * _R}
        auto end_time = std::chrono::steady_clock::now();

        other_time += std::chrono::duration<double>(end_time - start_time);

        start_time = std::chrono::steady_clock::now();
        // changed to RealType from here
        std::vector<RealType> factor_r(2*M[d]);  // real representation of factors to compute residual
        auto uv = model.moden(d);
        for (uint64_t i=0; i<(M[d]/2); i++) {
            factor_r[2*i] = uv[i].real();
            factor_r[2*i+1] = uv[i].imag();
            //factor_r[2*i] = RealType(0.0);
            //factor_r[2*i+1] = RealType(0.0);
        }
        std::vector<RealType> c_re(2*(2*(L-1)+1));  // real representation of the linear coef
        auto cl = model.linear();
        for (int64_t l = 0; l <= 2*(L-1); l++)
        {
            c_re[2*l] = cl[l].real();
            c_re[2*l+1] = cl[l].imag();
            //c_re[2*l] = RealType(1.0);
            //c_re[2*l+1] = RealType(0.0);
        }
        // uncomment to check residual here:
        // std::cout << "at the beginning  " <<  20 * (std::log10(model.residual_norm()) - std::log10(nrm)) << std::endl;
#pragma omp parallel
        {
            int t = omp_get_thread_num();
            // std::fill(B[t].begin(), B[t].begin() + JN, DataType(0.0));
            // std::fill(H[t].begin(), H[t].begin() + JN * JN, DataType(0.0));
            std::fill(B[t].begin(), B[t].begin() + JN, RealType(0.0));
            std::fill(H[t].begin(), H[t].begin() + JN * JN, RealType(0.0));   


#pragma omp for
            for (uint64_t k = 0; k < K; k += block_size)
            {
                uint64_t k_last = std::min(k + block_size, K);
                model.JacobianPart(d, k, k_last, J[t].begin(), 2*block_size);
                model.LinearPart(k, k_last, J[t].begin() + 2*block_size * M[d], 2*block_size);


                const uint64_t JM = k_last - k;

                // many changes -- a lot of changes in sizes, RealType of everything
                BLAS::gemv('N', 2*JM, M[d], RealType(1.0), J[t].data(), 2*block_size, factor_r.data(), 1, RealType(0.0), Re.data() + 2*k, 1);
                BLAS::gemv('N', 2*JM, 2*L, RealType(1.0), J[t].data() + 2*block_size * M[d], 2*block_size, c_re.data(), 1, RealType(1.0), Re.data() + 2*k, 1);
                BLAS::gemv('C', 2*JM, JN, RealType(1.0), J[t].data(), 2*block_size, rhs_real.data() + 2*k, 1, RealType(1.0), B[t].data(), 1);
                BLAS::herk('U', 'C', JN, 2*JM, RealType(1.0), J[t].data(), 2*block_size, RealType(1.0), H[t].data(), JN);
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
        for (uint64_t k = 0; k < K; k ++) {  // back to complex 
            R[k].real(Re[2*k]);
            R[k].imag(Re[2*k+1]);
        }
        info = LAPACK::potrf('U', JN, H[0].data(), JN);
        if (info != 0)
        {
            throw std::runtime_error("LAPACK::potrf failed");
        }
        LAPACK::potrs('U', JN, 1, H[0].data(), JN, B[0].data(), JN);

        // here solution is obtained
        model.update_new(d, B[0].begin());
        model.update_linear(B[0].begin() + M[d]);
        end_time = std::chrono::steady_clock::now();

        // uncomment to check residual here:
        // std::cout << "at the ending " << 20 * (std::log10(model.residual_norm()) - std::log10(nrm)) << std::endl;

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
                RealType add_err = 0;
                if (params.additional_metric)
                {
                    add_err = params.additional_metric(K, rhs, R.data());
                }
                BLAS::axpy(K, DataType(-1.0), rhs, 1, R.data(), 1);
                auto err = BLAS::nrm2(K, R.data(), 1);
                std::cout << i << ' ' << 20 * (std::log10(err) - std::log10(nrm)) << ' ';
                if (params.additional_metric)
                {
                    std::cout << add_err << ' ';
                }
                std::cout << (prev_err - err) / nrm
                          << ' ' << jac_gen_time.count() << ' ' << other_time.count() << std::endl;

                // if ((prev_err - err) <= std::max(abs_tol, rel_tol * nrm) || i == max_it)
                // {
                //     break;
                // }
                if (std::abs(prev_err - err) <= std::max(abs_tol, rel_tol * nrm) || i == max_it)
                {
                    break;
                }
                prev_err = err;
            }
        }
    }
}
#endif
