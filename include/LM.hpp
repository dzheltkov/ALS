#ifndef PBM2D_LM
#define PBM2D_LM

#include <cstdint>
#include <limits>
#include <algorithm>
#include <CXXBLAS.hpp>
#include <CXXLAPACK.hpp>
#include <iostream>
#include <chrono>

template<class DataType>
struct LMParams
{
    typedef decltype(std::abs(DataType(0.0))) RealType;
    uint64_t block_size = 4096;
    uint64_t max_it = 200;
    RealType rel_tol = 1000.0 * std::numeric_limits<RealType>::epsilon();
    RealType abs_tol = 0.0;
};

    template <class Model, class DataType>
void LM(Model &model, const DataType *rhs, const LMParams<DataType> &params = LMParams<DataType>())
{
    typedef decltype(std::abs(DataType(0.0))) RealType;
    const uint64_t D = model.dimensionality();
    const auto M = model.sizes();
    const uint64_t L = model.linear_size();

    const uint64_t K = model.length();

    const uint64_t sum_M = std::accumulate(M.begin(), M.begin() + D, L);

    const uint64_t block_size = std::min(params.block_size, K);

    const double rel_tol = params.rel_tol;
    const double abs_tol = params.abs_tol;
    uint64_t max_it = params.max_it;

    std::vector<DataType> J(block_size * sum_M);
    std::vector<DataType> B(sum_M);
    std::vector<DataType> H(sum_M * sum_M);
    std::vector<RealType> s(sum_M);
    std::vector<DataType> R(K);

    auto nrm = BLAS::nrm2(K, rhs, 1);

    bool left_to_right = true;

    std::chrono::duration<double> jac_gen_time;
    std::chrono::duration<double> herk_time;
    std::chrono::duration<double> other_time;
    for (uint64_t i = 0; i < max_it; i++)
    {
        auto start_time = std::chrono::steady_clock::now();
        std::fill(B.begin(), B.begin() + sum_M, DataType(0.0));
        std::fill(H.begin(), H.begin() + sum_M * sum_M, DataType(0.0));
        BLAS::copy(K, rhs, 1, R.data(), 1);
        auto end_time = std::chrono::steady_clock::now();

        other_time += std::chrono::duration<double>(end_time - start_time);

        for (uint64_t k = 0; k < K; k += block_size)
        {
            start_time = std::chrono::steady_clock::now();
            uint64_t k_last = std::min(k + block_size, K);
            uint64_t offset = 0;
            for (uint64_t d = 0; d < D; d++)
            {
                model.JacobianPart(d, k, k_last, J.begin() + offset, block_size);
                offset += M[d] * block_size;
            }
            model.LinearPart(k, k_last, J.begin() + offset, block_size);
            end_time = std::chrono::steady_clock::now();
            jac_gen_time += std::chrono::duration<double>(end_time - start_time);

            start_time = std::chrono::steady_clock::now();
            const uint64_t JM = k_last - k;

            BLAS::gemv('N', JM, M[0], DataType(-1.0), J.data(), block_size, model.mode(0), 1, DataType(1.0), R.data() + k, 1);
            BLAS::gemv('N', JM, L, DataType(-1.0), J.data() + offset, block_size, model.linear(), 1, DataType(1.0), R.data() + k, 1);

            BLAS::gemv('C', JM, sum_M, DataType(1.0), J.data(), block_size, R.data() + k, 1, DataType(1.0), B.data(), 1);

            BLAS::herk('U', 'C', sum_M, JM, RealType(1.0), J.data(), block_size, RealType(1.0), H.data(), sum_M);
            end_time = std::chrono::steady_clock::now();

            herk_time += std::chrono::duration<double>(end_time - start_time);
        }

        auto residual_norm = BLAS::nrm2(K, R.data(), 1);

        std::cout << i + 1 << ' ' << 20 * (std::log10(residual_norm) - std::log10(nrm)) << std::endl;
        auto prev_err = residual_norm;
        auto best_attempt = residual_norm;

        start_time = std::chrono::steady_clock::now();
        int info = 0;

        RealType scond, amax;

        info = LAPACK::poequb(sum_M, H.data(), sum_M, s.data(), scond, amax);

        bool equed = false;//LAPACK::laqhe('U', sum_M, H.data(), sum_M, s.data(), scond, amax);

        double lambda_min = std::numeric_limits<RealType>::epsilon();
        double lambda_max = RealType(1.0) / std::numeric_limits<RealType>::epsilon();
        if (!equed)
        {
            lambda_min *= amax * scond;
            lambda_max *= amax;
        }
        else
        {
            for (uint64_t i = 0; i < sum_M; i++)
            {
                B[i] *= s[i];
            }
        }

        std::vector<DataType> Hc(H.size());
        std::vector<DataType> Bc(B.size());
        std::vector<DataType> x(B.size());
        std::vector<DataType> new_x(B.size());

        uint64_t offset = 0;
        for (uint64_t d = 0; d < D; d++)
        {
            BLAS::copy(M[d], model.mode(d), 1, x.data() + offset, 1);
            offset += M[d];
        }
        BLAS::copy(L, model.linear(), 1, x.data() + offset, 1);
        std::vector<DataType> best_x = x;
        RealType best_lambda = 0;
        RealType best_alpha = 0;


        for (double lambda = lambda_min; lambda <= lambda_max; lambda *= 2)
        {
            BLAS::copy(H.size(), H.data(), 1, Hc.data(), 1);
            BLAS::copy(B.size(), B.data(), 1, Bc.data(), 1);

            for (uint64_t i = 0; i < sum_M; i++)
            {
                Hc[i * (sum_M + 1)] += lambda;
            }

            info = LAPACK::potrf('U', sum_M, Hc.data(), sum_M);
            if (info != 0)
            {
                continue;
            }
            LAPACK::potrs('U', sum_M, 1, Hc.data(), sum_M, Bc.data(), sum_M);
            if (equed)
            {
                for (uint64_t i = 0; i < sum_M; i++)
                {
                    Bc[i] *= s[i];
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

            auto alpha_best_norm = new_residual_norm;

            //std::cout << lambda << ' ' << 1.0 << ' ' << 20 * (std::log10(new_residual_norm) - std::log10(nrm)) << std::endl;

            if (new_residual_norm < residual_norm)
            {
                residual_norm = new_residual_norm;
                BLAS::copy(sum_M, new_x.data(), 1, best_x.data(), 1);
                best_lambda = lambda;
                best_alpha = 1.0;
            }

            RealType alpha = 1;
            while(true)
            {
                alpha /= 2;
                BLAS::copy(sum_M, x.data(), 1, new_x.data(), 1);
                BLAS::axpy(sum_M, DataType(alpha), Bc.data(), 1, new_x.data(), 1);
                uint64_t offset = 0;
                for (uint64_t d = 0; d < D; d++)
                {
                    model.update(d, new_x.begin() + offset);
                    offset += M[d];
                }
                model.update_linear(new_x.data() + offset);

                auto new_residual_norm = model.residual_norm();
                //std::cout << lambda << ' ' << alpha << ' ' << 20 * (std::log10(new_residual_norm) - std::log10(nrm)) << std::endl;

                if (new_residual_norm < residual_norm)
                {
                    residual_norm = new_residual_norm;
                    BLAS::copy(sum_M, new_x.data(), 1, best_x.data(), 1);
                    best_lambda = lambda;
                    best_alpha = alpha;
                }
                if (new_residual_norm < alpha_best_norm)
                {
                    alpha_best_norm = new_residual_norm;
                }
                else
                {
                    break;
                }
            }
            alpha = 1;
            while(true)
            {
                alpha *= 2;
                BLAS::copy(sum_M, x.data(), 1, new_x.data(), 1);
                BLAS::axpy(sum_M, DataType(alpha), Bc.data(), 1, new_x.data(), 1);
                uint64_t offset = 0;
                for (uint64_t d = 0; d < D; d++)
                {
                    model.update(d, new_x.begin() + offset);
                    offset += M[d];
                }
                model.update_linear(new_x.data() + offset);

                auto new_residual_norm = model.residual_norm();
                //std::cout << lambda << ' ' << alpha << ' ' << 20 * (std::log10(new_residual_norm) - std::log10(nrm)) << std::endl;

                if (new_residual_norm < residual_norm)
                {
                    residual_norm = new_residual_norm;
                    BLAS::copy(sum_M, new_x.data(), 1, best_x.data(), 1);
                    best_lambda = lambda;
                    best_alpha = alpha;
                }
                if (new_residual_norm < alpha_best_norm)
                {
                    alpha_best_norm = new_residual_norm;
                }
                else
                {
                    break;
                }
            }
        }
        offset = 0;
        for (uint64_t d = 0; d < D; d++)
        {
            model.update(d, best_x.begin() + offset);
            offset += M[d];
        }
        model.update_linear(best_x.data() + offset);
        std::cout << (prev_err - residual_norm) / nrm << ' ' << best_lambda << ' ' << best_alpha << std::endl << std::endl;

        if ((prev_err - residual_norm) <= std::max(abs_tol, rel_tol * nrm))
        {
            break;
        }
    }
    std::cout << 20 * (std::log10(model.residual_norm()) - std::log10(nrm)) << std::endl;
}
#endif
