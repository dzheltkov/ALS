#ifndef PBM2D_LM
#define PBM2D_LM

#include <cstdint>
#include <limits>
#include <algorithm>
#include <CXXBLAS.hpp>
#include <CXXLAPACK.hpp>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <queue>

template<class DataType>
struct LMParams
{
    typedef decltype(std::abs(DataType(0.0))) RealType;
    uint64_t block_size = 64;
    uint64_t max_it = 100000;
    int64_t verbose = 1;
    int64_t lambda_check = 10;
    RealType rel_tol = 1024.0 * std::numeric_limits<RealType>::epsilon();
    RealType abs_tol = 0.0;
    std::function<std::complex<double> (int64_t, const std::complex<double> *, const std::complex<double> *)> additional_metric = nullptr;
};

    template <class Model, class DataType>
void LM(Model &model, const DataType *rhs, const LMParams<DataType> &params = LMParams<DataType>())
{
    typedef decltype(std::abs(DataType(0.0))) RealType;
    const uint64_t D = model.dimensionality();
    const uint64_t K = model.length();
    const uint64_t L = model.linear_size();
    const auto M = model.sizes();
    const uint64_t sum_M = std::accumulate(M.begin(), M.begin() + D, L);

    const uint64_t block_size = std::min(params.block_size, K);

    const double rel_tol = params.rel_tol;
    const double abs_tol = params.abs_tol;
    uint64_t max_it = params.max_it;

    std::vector<std::vector<DataType> > J(omp_get_max_threads(), std::vector<DataType>(block_size * sum_M));
    std::vector<std::vector<DataType> > H(omp_get_max_threads(), std::vector<DataType>(sum_M * sum_M));
    std::vector<std::vector<DataType> > B(omp_get_max_threads(), std::vector<DataType>(sum_M));
    std::vector<DataType> prediction(K);
    std::vector<DataType> R(K);
    std::vector<RealType> s(sum_M); // scaling параметры для уменьшения числа обусловленности


    auto nrm = BLAS::nrm2(K, rhs, 1);

    std::chrono::duration<double> jac_gen_time;
    std::chrono::duration<double> other_time;
    RealType last_best_lambda;

    for (uint64_t i = 0; i < max_it; i++)
    {
        auto start_time = std::chrono::steady_clock::now();
        BLAS::copy(K, rhs, 1, R.data(), 1);
        auto end_time = std::chrono::steady_clock::now();
        other_time += std::chrono::duration<double>(end_time - start_time);
#pragma omp parallel
        {
            int t = omp_get_thread_num();
            std::fill(B[t].begin(), B[t].begin() + sum_M, DataType(0.0));
            std::fill(H[t].begin(), H[t].begin() + sum_M * sum_M, DataType(0.0));
#pragma omp for
            for (uint64_t k = 0; k < K; k += block_size)
            {
                //std::fill(J[t].begin(), J[t].begin() + sum_M * block_size, DataType(0.0));
                uint64_t k_last = std::min(k + block_size, K);

                uint64_t offset = 0;
                for (uint64_t d = 0; d < D; d++)
                {
                    model.JacobianPart(d, k, k_last, J[t].begin() + offset, block_size);
                    offset += M[d] * block_size;
                }
                model.LinearPart(k, k_last, J[t].begin() + offset, block_size);

                const uint64_t JM = k_last - k;
                BLAS::gemv('N', JM, M[0], DataType(1.0), J[t].data(), block_size, model.mode(0), 1, DataType(0.0), prediction.data() + k, 1);
                BLAS::gemv('N', JM, L, DataType(1.0), J[t].data() + offset, block_size, model.linear(), 1, DataType(1.0), prediction.data() + k, 1);

                BLAS::gemv('N', JM, M[0], DataType(-1.0), J[t].data(), block_size, model.mode(0), 1, DataType(1.0), R.data() + k, 1);
                BLAS::gemv('N', JM, L, DataType(-1.0), J[t].data() + offset, block_size, model.linear(), 1, DataType(1.0), R.data() + k, 1);

                BLAS::gemv('C', JM, sum_M, DataType(1.0), J[t].data(), block_size, R.data() + k, 1, DataType(1.0), B[t].data(), 1);
                BLAS::herk('U', 'C', sum_M, JM, RealType(1.0), J[t].data(), block_size, RealType(1.0), H[t].data(), sum_M);
            }
#pragma omp for nowait
            for (uint64_t j = 0; j < sum_M * sum_M; j++)
            {
                for (int64_t l = 1; l < omp_get_num_threads(); l++)
                {
                    H[0][j] += H[l][j];
                }
            }
#pragma omp for nowait
            for (uint64_t j = 0; j < sum_M; j++)
            {
                for (int64_t l = 1; l < omp_get_num_threads(); l++)
                {
                    B[0][j] += B[l][j];
                }
            }
        }

        end_time = std::chrono::steady_clock::now();
        jac_gen_time += std::chrono::duration<double>(end_time - start_time);

        auto residual_norm = BLAS::nrm2(K, R.data(), 1);
        std::cout << i + 1 << ' ' << 20 * (std::log10(residual_norm) - std::log10(nrm)) << std::endl;
        auto prev_err = residual_norm;

        int info = 0;

        RealType scond, amax;
        info = LAPACK::poequb(sum_M, H[0].data(), sum_M, s.data(), scond, amax);
        bool equed = LAPACK::laqhe('U', sum_M, H[0].data(), sum_M, s.data(), scond, amax);

        if (equed) {
            for (uint64_t j = 0; j < sum_M; j++)
            {
                B[0][j] *= s[j];
            }
        }  
        info = 0;
        start_time = std::chrono::steady_clock::now();

        std::vector<DataType> Hc(H[0].size());
        std::vector<DataType> Bc(B[0].size());
        std::vector<DataType> x(B[0].size());
        std::vector<DataType> new_x(B[0].size());

        uint64_t offset = 0;
        for (uint64_t d = 0; d < D; d++)
        {
            BLAS::copy(M[d], model.mode(d), 1, x.data() + offset, 1);
            offset += M[d];
        }
        BLAS::copy(L, model.linear(), 1, x.data() + offset, 1);

        std::vector<DataType> local_best_x = x;

        double local_best_lambda;
        double local_residual_norm = RealType(1.0) / std::numeric_limits<RealType>::epsilon(); // лучшая из проверенных лямбд 
        double checked_left_boundary = RealType(1.0) / std::numeric_limits<RealType>::epsilon(), 
            checked_right_boundary = std::numeric_limits<RealType>::epsilon();
        bool improve_resid = false;
        int count = 0;

        std::queue <double> lambda_q;

        if (i == 0) {
            for (double lambda = std::numeric_limits<RealType>::epsilon(); 
                    lambda <= RealType(1.0) / std::numeric_limits<RealType>::epsilon(); lambda *= 512) {
                lambda_q.push(lambda);
            }
        } else {
            lambda_q.push(local_best_lambda);
            lambda_q.push(local_best_lambda / 2);
            lambda_q.push(local_best_lambda * 2);
        }

        while (true)
        {
            count++;
            double lambda = lambda_q.front();
            lambda_q.pop();
            checked_left_boundary = std::min(checked_left_boundary, lambda);
            checked_right_boundary = std::max(checked_right_boundary, lambda);
            
            BLAS::copy(H[0].size(), H[0].data(), 1, Hc.data(), 1);
            BLAS::copy(B[0].size(), B[0].data(), 1, Bc.data(), 1);

            for (uint64_t j = 0; j < sum_M; j++)
            {
                Hc[j * (sum_M + 1)] *= 1 + lambda;
            }

            info = LAPACK::potrf('U', sum_M, Hc.data(), sum_M);
            if (info != 0)
            {
                continue;
            }
            LAPACK::potrs('U', sum_M, 1, Hc.data(), sum_M, Bc.data(), sum_M);

            if (equed) {
                for (uint64_t j = 0; j < sum_M; j++)
                {
                    Bc[j] *= s[j];
                }
            }

            BLAS::copy(sum_M, x.data(), 1, new_x.data(), 1);
            BLAS::axpy(sum_M, DataType(1.0), Bc.data(), 1, new_x.data(), 1);
            
            uint64_t offset = 0;
            for (uint64_t d = 0; d < D; d++)
            {
                model.update(d, new_x.begin() + offset);
                offset += M[d];
            }
            model.update_linear(new_x.data() + offset);

            auto new_residual_norm = model.residual_norm();

            std::cout << lambda << ' ' << 20 * (std::log10(new_residual_norm) - std::log10(nrm)) << std::endl;

            if (new_residual_norm < residual_norm)
            {
                residual_norm = new_residual_norm;
                improve_resid = true;
            }
            if (new_residual_norm < local_residual_norm)
            {
                local_residual_norm = new_residual_norm;
                BLAS::copy(sum_M, new_x.data(), 1, local_best_x.data(), 1);
                local_best_lambda = lambda;
            }

            if (lambda_q.size() == 0) {
                if (improve_resid == true) {
                    if (local_best_lambda > last_best_lambda) {
                        if (local_best_lambda * 2 <= RealType(1.0) / std::numeric_limits<RealType>::epsilon()) {
                            lambda_q.push(local_best_lambda * 2);
                        }
                    } else if (local_best_lambda < last_best_lambda) {
                        if (local_best_lambda / 2 >= std::numeric_limits<RealType>::epsilon()) {
                            lambda_q.push(local_best_lambda / 2);
                        }
                    } else {
                        break;
                    }
                    last_best_lambda = local_best_lambda;
                } else {
                    if (checked_left_boundary / 2 >= std::numeric_limits<RealType>::epsilon()) {
                        lambda_q.push(checked_left_boundary / 2);
                    }
                    if (checked_right_boundary * 2 <= RealType(1.0) / std::numeric_limits<RealType>::epsilon()) {
                        lambda_q.push(checked_right_boundary * 2);
                    }
                }
                if (lambda_q.size() == 0) {
                    break;
                }
            }
            if (count >= params.lambda_check) {
                break;
            }
        }
        if (improve_resid) {
            residual_norm = local_residual_norm;
        }
        offset = 0;
        for (uint64_t d = 0; d < D; d++)
        {
            model.update(d, local_best_x.begin() + offset);
            offset += M[d];
        }
        model.update_linear(local_best_x.data() + offset);
        last_best_lambda = local_best_lambda;
        end_time = std::chrono::steady_clock::now();
        other_time += std::chrono::duration<double>(end_time - start_time);
        std::complex<double> add_err = 0;

        if (params.additional_metric)
        {
            add_err = params.additional_metric(model.length(), reinterpret_cast<const std::complex<double>*>(rhs), reinterpret_cast<std::complex<double>*>(prediction.data()));
        }

        if (params.verbose) {
            std::cout << "Iter #" << i + 1 << " NMSE: " << 20 * (std::log10(residual_norm) - std::log10(nrm)) << " " 
                << " fall: " << (prev_err - residual_norm) / nrm << " BER: " << add_err.real() << " best lambda: " << local_best_lambda << "  Jacobian: " << jac_gen_time.count() << " Other: " << other_time.count() << std::endl << std::endl;
        }

        model.logging_error(i, 20 * (std::log10(residual_norm) - std::log10(nrm)),
            (prev_err - residual_norm) / nrm, jac_gen_time.count(),
            other_time.count(), add_err);

        if ((prev_err - residual_norm) <= std::max(abs_tol, rel_tol * nrm))
        {
            break;
        }
    }
    std::cout << 20 * (std::log10(model.residual_norm()) - std::log10(nrm)) << std::endl;
}
#endif



