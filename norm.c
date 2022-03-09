//******************************************************************************
// Perform norm of arrays across MPI ranks and GPUs
// MPI ranks are assigned to available devices
//******************************************************************************
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

/*----------------------------------------------------------------------------*/

void init_data(double* h_U, int h_N, MPI_Comm comm) {

  int rank;
  MPI_Comm_rank(comm, &rank);

  for (int i=0; i<h_N; i++) { h_U[i] = i + 1000.0*rank; }
}

/*----------------------------------------------------------------------------*/

void set_myRank_deviceId(MPI_Comm comm) {

  int rank;
  MPI_Comm_rank(comm, &rank);

  int device_id = rank % omp_get_num_devices();
  omp_set_default_device(device_id);
}

/*----------------------------------------------------------------------------*/

double NormMPIReduce(double *h_norm, int N, MPI_Comm comm) {

  double norm;
  MPI_Allreduce(h_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
  int Ntot;
  MPI_Allreduce(&N, &Ntot, 1, MPI_INT, MPI_SUM, comm);
  
  return sqrt(norm)/Ntot;
}

/*----------------------------------------------------------------------------*/

void NormKernelOMT(const int N,
                   const double* __restrict__ U,
                         double* __restrict__ norm2) {

  double norm=0;

  #pragma omp target teams distribute parallel for \
          map(to: N, U[0:N]) map(tofrom: norm)reduction(+:norm)
  for (int id=0; id < N; id++) {
    norm += U[id] * U[id];
  }

  *norm2 = norm;
}

/*----------------------------------------------------------------------------*/

double NormOMT(double *U, int N, MPI_Comm comm) {

  double h_norm=0;
  NormKernelOMT(N, U, &h_norm);

  return NormMPIReduce(&h_norm, N, comm);
}

/*----------------------------------------------------------------------------*/

void NormKernelOMP(const int N,
                   const double* __restrict__ U,
                         double* __restrict__ norm2) {

  double norm=0;

  #pragma omp parallel for reduction(+:norm)
  for (int id=0; id < N; id++) {
    norm += U[id] * U[id];
  }

  *norm2 = norm;
}

/*----------------------------------------------------------------------------*/

double NormOMP(double *U, int N, MPI_Comm comm) {

  double h_norm=0;
  NormKernelOMP(N, U, &h_norm);

  return NormMPIReduce(&h_norm, N, comm);
}

/*----------------------------------------------------------------------------*/

void NormKernelMPI(const int N,
                   const double* __restrict__ U,
                         double* __restrict__ norm2) {

  double norm=0;
  for (int id=0; id < N; id++) {
    norm += U[id] * U[id];
  }

  *norm2 = norm;
}

/*----------------------------------------------------------------------------*/

double NormMPI(double *U, int N, MPI_Comm comm) {

  double h_norm=0;
  NormKernelMPI(N, U, &h_norm);

  return NormMPIReduce(&h_norm, N, comm);
}

/*----------------------------------------------------------------------------*/

int main()
{
  // MPI initializations
  MPI_Init(NULL, NULL);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (size > omp_get_num_devices()) {
    if (rank==0)
      printf("*** Error: mpi size=%d > total gpus=%d ***\n",size, omp_get_num_devices());
    return EXIT_FAILURE;
  }

  // Set host input/output arrays
  // Define local mpi-rank arrays: all ranks have same sized (h_N) local arrays
  int h_N = 10;
  double h_U[h_N];
  init_data(h_U, h_N, comm);

  // Pick a gpu device_id for this rank
  set_myRank_deviceId(comm);

  // My rank device_id operations
  double norm_mpi = NormMPI(h_U, h_N, comm);
  double norm_omp = NormOMP(h_U, h_N, comm);
  double norm_omt = NormOMT(h_U, h_N, comm);

  if (rank==0)
      printf(" norm_mpi = %0.16f\n norm_omp = %0.16f\n norm_omt = %0.16f\n",norm_mpi, norm_omp, norm_omt);

  MPI_Finalize();

  // Error check and exit
  int err =  (norm_omt == norm_omp) && (norm_omt == norm_mpi) ? 0: 1;
  if (!err) {
    if (rank==0)
      printf("Success\n");
    return EXIT_SUCCESS;
  } else {
    if (rank==0)
      printf("Failure!\n");
    return EXIT_FAILURE;
  }

  //return EXIT_SUCCESS;
}
