#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>
#include "MatrixVector.h"

int main (int argc, char* argv[])
{
    int world_size;
    int world_rank;
    int world_root = 0;  // root-process

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* 2D-heat equation in [0,1]x[0,1] */
    int Nx = 100;
    int Ny = 100;

    double T = 1.0;
    double X = 1.0;
    double Y = 1.0;

    auto U_exact = [](double x, double y, double t){
        return std::sin(M_PI * x) * std::sin(M_PI * y) * std::exp(-t);
    };
    double lambda = 0.5 / M_PI / M_PI;

    int iteration = 0;
    double t  = 0.0;
    double dx = X / Nx;
    double dy = Y / Ny;

    /* Courant–Friedrichs–Lewy condition */
    double dt = 0.1 * std::min(dx, dy) * std::min(dx, dy) / lambda;

    Vector meshX(Nx+1, X);
    Vector meshY(Ny+1, Y);
    for (int i = 0; i < Nx; ++i){
        meshX[i] = 0.0 + dx * i;
    }
    for (int i = 0; i < Ny; ++i){
        meshY[i] = 0.0 + dy * i;
    }
    Matrix U(Nx+1, Ny+1);
    for (int ix = 0; ix <= Nx; ++ix){
        for (int iy = 0; iy <= Ny; ++iy){
            U.at(ix, iy) = U_exact(meshX[ix], meshY[iy], 0.0);
        }
    }

    /* MPI splitting data */
    int *block_size        = new int[world_size];
    int *block_index       = new int[world_size];
    int *block_matrix_size = new int[world_size];
    int *block_matrix_index= new int[world_size];

    DefineSplittingParameters(block_index, block_size, Nx, MPI_COMM_WORLD);
    for (int i = 0; i < world_size; ++i){
        block_matrix_size[i] = block_size[i] * (Ny+1);
        block_matrix_index[i] = block_index[i] * (Ny+1);
    }

    Matrix U_next(block_size[world_rank], Ny+1);
    t += dt;
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed_time = -MPI_Wtime();

    while (t < T){
        for (int ix = block_index[world_rank]; ix < block_index[world_rank] + block_size[world_rank]; ++ix){
            int local_ix = ix - block_index[world_rank];
            for (int iy = 0; iy < Ny + 1; ++iy){
                if ((ix == 0) || (ix == Nx) || (iy == 0) || (iy == Ny)){
                    U_next.at(local_ix, iy) = U_exact(meshX[ix], meshY[iy], t);
                }
                else{
                    U_next.at(local_ix, iy) = U.at(ix, iy) +
                            lambda * dt *
                            ((U.at(ix+1, iy) - 2*U.at(ix, iy) + U.at(ix-1, iy))/dx/dx +
                             (U.at(ix, iy+1) - 2*U.at(ix, iy) + U.at(ix, iy-1))/dy/dy);
                }
            }
        }

        MPI_Allgatherv(
                U_next.GetData(),
                block_matrix_size[world_rank],
                MPI_DOUBLE,
                U.GetData(),
                block_matrix_size,
                block_matrix_index,
                MPI_DOUBLE,
                MPI_COMM_WORLD);
        t += dt;
        ++iteration;
    }
    elapsed_time += MPI_Wtime();

    t -= dt;
    if (world_rank == world_root)
    {
        Matrix error = U;
        for (int ix = 0; ix <= Nx; ++ix)
        {
            for (int iy = 0; iy <= Ny; ++iy)
            {
                error.at(ix, iy) -= U_exact(meshX[ix], meshY[iy], t);
            }
        }
        double rel_error = error.l2Norm();
        std::cout << "-- 2D-heat equation solver --\n"
                  << "   Nx = " << Nx << '\n'
                  << "   Ny = " << Ny << '\n'
                  << "   t  = " << t << '\n'
                  << "   time_iter. = " << iteration << '\n'
                  << "   rel_error = " << rel_error << "\n"
                  << "   time = " << elapsed_time << std::endl;

        std::string filename = "output/MPI_proc_" + std::to_string(world_size) +
                               "_Nx_" + std::to_string(Nx) + ".txt";
        std::ofstream out(filename);
        out << "-- 2D-heat equation solver --\n"
            << "   Nx = " << Nx << '\n'
            << "   Ny = " << Ny << '\n'
            << "   t  = " << t << '\n'
            << "   time_iter. = " << iteration << '\n'
            << "   rel_error = " << rel_error << "\n"
            << "   time = " << elapsed_time << std::endl;
    }

    MPI_Finalize();
    return 0;
}
