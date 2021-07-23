/**
 * @file serial_omp_vec_time.c
 * gcc ./serial_omp_vec_time.c -o serial_omp_vec_time -mavx2 -fopenmp
 *
 * @brief 逐次、OpenMPによるスレッド処理、AVX2での比較
 * 逐次のみ(SERIAL使用、OMPはコメントアウト)
 *
 * OpenMPで並列化(SERIALとOMP使用)\n
 * AVX2でベクトル化(どちらもコメントアウト)\n
 * OpenMP+AVX2(OMP使用、SERIALコメントアウト)\n
 * @author E.Okazaki
 */

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

//! 8要素のfloatを32バイト境界でアライメントされた型をfloat8とする
typedef float float8[8] __attribute__((aligned(32)));

#define SERIAL
#define OMP

#define ROW 65536
#define COL  sizeof(float8) / sizeof(float)
#define LOOP_NUM 10000

/**
 * @brief 行列aに値bを乗算して、それをdestに代入する関数
 */
void multiply(float8* dest, float8* a, float b)
{
    printf("%d行%ld列の要素全てに%.fを乗算することを%d回ループします。\n", ROW, COL, b, LOOP_NUM);
#ifdef OMP
    // OMPが定義されているならOpenMPのfor構文を使う
    #pragma omp parallel for
#endif

    for (int loop_counter = 0; loop_counter < LOOP_NUM; loop_counter++) {
#ifdef SERIAL
        // SERIALが定義されているなら1つずつ処理する
        for (int i = 0; i < ROW; i += 1) {
            for (int j = 0; j < COL; j += 1) {
                dest[i][j] = a[i][j] * b;
            }
        }
#else
        // SERIALが定義されていないならベクトル化処理を行う
        // bを8つとして、YMMへロード
        __m256 vb = _mm256_broadcast_ss((float*)&b);
        for (int i = 0; i < ROW; i += 1) {
            // a[i]をYMMへロード
            __m256 va = _mm256_load_ps((float*)&a[i]);
            // va,vbを乗算
            __m256 vmul = _mm256_mul_ps(va, vb);
            // vmulをdest[i]へストア
            _mm256_store_ps((float*)&dest[i], vmul);
        }
#endif
    }
}

int main()
{
    float8 a[ROW], res[ROW];
    // シードをUNIX時間にする
    srand((unsigned int)time(NULL));
    // 適当な値を行列に代入していく
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            a[i][j] = (float)rand();
        }
    }
    // 掛ける値
    float r = (float)rand();
    // 開始時刻を取得
    double start = omp_get_wtime();
    multiply(&res[0], &a[0], r);
    // 終了時刻を取得
    double end = omp_get_wtime();
    // 差分から計算にかかった時間を算出
    printf("経過時間: %f秒\n", end - start);
}
