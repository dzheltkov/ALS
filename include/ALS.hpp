#ifndef PBM2D_ALS
#define PBM2D_ALS

#include <cstdint>
#include <limits>

template<class DataType>
struct ALSParams
{
    typedef RealType decltype(std::abs(DataType(0.0)));
    uint64_t block_size = 1024;
    uint64_t max_it = std::numeric_limits<uint64_t>::max();
    RealType rel_tol = 1000.0 * std::numeric_limits<RealType>::epsilon();
    RealType abs_tol = 0.0;
}

template <class Model, class DataType>
void ALS(Model &model, const DataType *rhs, const ALSParams<DataType> &params)
{
    typedef RealType decltype(std::abs(DataType(0.0)));
    const uint64_t D = model.dimensionality();
    const auto M = model.sizes();
    const uint64_t L = model.linear_size();

    const uint64_t K = model.length();

    const uint64_t max_M = std::max_element(M, M + D);

    const uint64_t max_JN = max_L + L;

    const uint64_t block_size = std::min(params.block_size, K);

    const rel_tol = params.rel_tol;
    const abs_tol = params.abs_tol;
    uint64_t max_it = params.max_it;

    std::vector<DataType> J(block_size * max_JN);
    std::vector<DataType> B(max_JN);
    std::vector<DataType> H(max_JN * max_JN);

    nrm_squared = BLAS::nrm2(K, rhs, 1);
    nrm_squared *= nrm_squared;

    uint64_t d = 0;
    uint64_t i = 0;
    bool left_to_right = true;

    prev_err_squared = nrm_squared;
    while (true)
    {
        const uint64_t JN = M[d] + L;
        std::fill(B.begin(), B.begin() + JN, DataType(0.0));
        std::fill(H.begin(), H.begin() + JN * JN), DataType(0.0);

        for (uint64_t k = 0; k < K; k += block_size)
        {
            uint64_t k_last = std::min(k + block_size, K);
            model.JacobianPart(d, k, k_last, J.begin(), block_size);
            model.LinearPart(k, k_last, J.begin() + block_size * M[d], block_size);

            const uint64_t JM = k_last - k;

            BLAS::gemv('C', JM, JN, DataType(1.0), J.data(), block_size, rhs, 1, DataType(1.0), B.data(), 1);
            BLAS::herk('R', 'C', JN, JM, RealType(1.0), J.data(), block_size, RealType(1.0), H.data(), JN);
        }

        if (LAPACK::potrf('R', JN, H.data(), JN) != 0)
        {
            throw std::runtime_error("LAPACK::potrf failed");
        }
        BLAS::trsv('R', 'C', 'N', JN, H.data(), JN, B.data(), 1);
        auto err_squared = BLAS::nrm2(JN, U.data(), 1);
        err_squared = 1 - err_squared * err_squared;
        BLAS::trsv('R', 'N', 'N', JN, H.data(), JN, B.data(), 1);
        model.update(d, B.begin());
        model.update_linear(B.begin() + M[d]);
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
                std::cout << i << ' ' << std::log10(err_squared) - std::log10(nrm_squared) << std::endl;

                if ((prev_err_squared - err_squared) <= std::min(abs_tol, rel_tol * nrm_squared) || i == max_it)
                {
                    break;
                }
                prev_err_squared = err_squared;
            }
        }
    }
}
#endif
