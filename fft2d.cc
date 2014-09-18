// Distributed two-dimensional Discrete FFT transform
// YOUR NAME HERE
// ECE8893 Project 1

// Do the 2D transform here.
  // 1) Use the InputImage object to read in the Tower.txt file and
  //    find the width/height of the input image.
  // 2) Use MPI to find how many CPUs in total, and which one
  //    this process is
  // 3) Allocate an array of Complex object of sufficient size to
  //    hold the 2d DFT results (size is width * height)
  // 4) Obtain a pointer to the Complex 1d array of input data
  // 5) Do the individual 1D transforms on the rows assigned to your CPU
  // 6) Send the resultant transformed values to the appropriate
  //    other processors for the next phase.
  // 6a) To send and receive columns, you might need a separate
  //     Complex array of the correct size.
  // 7) Receive messages from other processes to collect your columns
  // 8) When all columns received, do the 1D transforms on the columns
  // 9) Send final answers to CPU 0 (unless you are CPU 0)
  //   9a) If you are CPU 0, collect all values from other processors
  //       and print out with SaveImageData().

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>
#include <cstdlib>

#include "Complex.h"
#include "InputImage.h"

using namespace std;

void Transpose(Complex* input, int width, int height, Complex* output) ;
void Transform1D_row(Complex* input, int width, Complex* output) ;
void Transform1D_myRows(Complex* in, int width, int myRows_start, int myRows, Complex* out) ;
void Transform1D_row_Inverse(Complex* input, int width, Complex* output) ;
void Transform1D_myRows_Inverse(Complex* in, int width, int myRows_start, int myRows, Complex* out) ;

int nCPUs, rank ;
int rc ;

void Transform2D(const char* inputFN) 
{ 
  InputImage image(inputFN);  // Create the helper object for reading the image
  // Step (1) in the comments is the line above.
  // Your code here, steps 2-9
  MPI_Comm_size(MPI_COMM_WORLD,&nCPUs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  int height = image.GetHeight();
  int width = image.GetWidth();
  Complex* h = image.GetImageData();
  const int myRows = width/nCPUs;
  const int startRow = myRows*rank;
  const int myRows_len = myRows * width ;

  Complex* H ;
  Complex* after1D ;
  Complex* before2D ;
  Complex* after2D ;
  Complex* before2D_Transpose ;
  Complex* collect2D ;

  after1D = new Complex[width * height] ;
  collect2D = new Complex[width * height] ;
  before2D_Transpose = new Complex[width * height] ;
  before2D = new Complex[width * height] ;
  after2D = new Complex[width * height] ; 
  H = new Complex[width * height] ;


  Transform1D_myRows(h, width, startRow, myRows, after1D) ;

  // send after1D to all
  for (int cpuN = 0; cpuN < nCPUs; ++cpuN) {
        if (cpuN != rank)
    {
    MPI_Request request;
    rc = MPI_Isend(&after1D[myRows_len * rank], myRows_len * sizeof(Complex) , MPI_CHAR, cpuN, 0, MPI_COMM_WORLD, &request);
    if (rc != MPI_SUCCESS)
    {
      cout << "Rank " << rank
          << " send failed, rc " << rc << endl;
      MPI_Finalize();
      exit(1);
     }
     //MPI_Status status;
     // MPI_Wait(&request, &status);
    }
  }

  memcpy((char *)&before2D[myRows_len * rank], (char *)&after1D[myRows_len * rank], myRows_len * sizeof(Complex));


    for (int cpuN = 0; cpuN < nCPUs; ++cpuN) {
    if (cpuN != rank)
    {
    MPI_Status status;
    rc = MPI_Recv(&before2D[myRows_len * cpuN], myRows_len * sizeof(Complex) , MPI_CHAR, cpuN, 0, MPI_COMM_WORLD, &status);
    if (rc != MPI_SUCCESS)
    {
      cout << "Rank " << rank
          << " send failed, rc " << rc << endl;
      MPI_Finalize();
      exit(1);
    }
  }
  }
    // Every CPU receive 1D
  Transpose(before2D, width, height, before2D_Transpose) ;

  Transform1D_myRows(before2D_Transpose, width, startRow, myRows, after2D) ;

  MPI_Request request;
  rc = MPI_Isend(&after2D[myRows_len * rank], myRows_len * sizeof(Complex) , MPI_CHAR, 0, 0, MPI_COMM_WORLD, &request);
  if (rc != MPI_SUCCESS)
  {
    cout << "Rank " << rank
        << " send failed, rc " << rc << endl;
    MPI_Finalize();
    exit(1);
  }
// after2D is transposed
    
  if (rank == 0) {
    for (int cpuN = 1; cpuN < nCPUs; ++cpuN) {
      MPI_Status status;
      rc = MPI_Recv(&collect2D[myRows_len * cpuN],myRows_len * sizeof(Complex), MPI_CHAR, cpuN , 0, MPI_COMM_WORLD, &status);
      if (rc != MPI_SUCCESS)
      {
        cout << "Rank " << rank
            << " send failed, rc " << rc << endl;
        MPI_Finalize();
        exit(1);
      }

    }
  memcpy((char *)&collect2D[myRows_len * rank], (char *)&after2D[myRows_len * rank], myRows_len * sizeof(Complex));


    Transpose(collect2D, width, height, H);

    image.SaveImageData("MyAfter2d.txt",H,width,height);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  /************************************  MyAfter2d.txt  DONE**************************************************************/

    delete[] after1D ;
    delete[] before2D ;
    delete[] before2D_Transpose ;

  Complex* after1D_inverse ;
  Complex* before1D_inverse ;
  Complex* before2D_inverse_Transpose ;
  Complex* after2D_inverse ;
  Complex* collect2D_inverse ;
  Complex* before1D_inverse_Transpose ;
  Complex* before2D_inverse ;
  before1D_inverse = new Complex[width * height] ;
  before1D_inverse_Transpose = new Complex[width * height] ;
  collect2D_inverse = new Complex[width * height] ;
  before2D_inverse = new Complex[width * height] ;
  before2D_inverse_Transpose = new Complex[width * height] ;
  after2D_inverse = new Complex[width * height] ;
  after1D_inverse = new Complex[width * height] ;

  for (int cpuN = 0; cpuN < nCPUs; ++cpuN) {
    if (cpuN != rank)
    {
    MPI_Request request;
    rc = MPI_Isend(&after2D[myRows_len * rank], myRows_len * sizeof(Complex) , MPI_CHAR, cpuN, 0, MPI_COMM_WORLD, &request);
    if (rc != MPI_SUCCESS)
    {
      cout << "Rank " << rank
          << " send failed, rc " << rc << endl;
      MPI_Finalize();
      exit(1);
     }
    }
  }

    for (int cpuN = 0; cpuN < nCPUs; ++cpuN) {
          if (cpuN != rank)
    {
    MPI_Status status;
    rc = MPI_Recv(&before1D_inverse[myRows_len * cpuN], myRows_len * sizeof(Complex) , MPI_CHAR, cpuN, 0, MPI_COMM_WORLD, &status);
    if (rc != MPI_SUCCESS)
    {
      cout << "Rank " << rank
          << " send failed, rc " << rc << endl;
      MPI_Finalize();
      exit(1);
    }
    }
  }
  memcpy((char *)&before1D_inverse[myRows_len * rank], (char *)&after2D[myRows_len * rank], myRows_len * sizeof(Complex));


  //  Transpose(before1D_inverse, width, height, before1D_inverse_Transpose) ;

   Transform1D_myRows_Inverse(before1D_inverse, width, startRow, myRows, after1D_inverse) ;  // after1D_inverse is transposed

    // send after1D_inverse to all
  for (int cpuN = 0; cpuN < nCPUs; ++cpuN) {
        if (cpuN != rank)
    {
    MPI_Request request;
    rc = MPI_Isend(&after1D_inverse[myRows_len * rank], myRows_len * sizeof(Complex) , MPI_CHAR, cpuN, 0, MPI_COMM_WORLD, &request);
    if (rc != MPI_SUCCESS)
    {
      cout << "Rank " << rank
          << " send failed, rc " << rc << endl;
      MPI_Finalize();
      exit(1);
     }

    }
  }

    for (int cpuN = 0; cpuN < nCPUs; ++cpuN) {
          if (cpuN != rank)
    {
    MPI_Status status;
    rc = MPI_Recv(&before2D_inverse[myRows_len * cpuN], myRows_len * sizeof(Complex) , MPI_CHAR, cpuN, 0, MPI_COMM_WORLD, &status);
    if (rc != MPI_SUCCESS)
    {
      cout << "Rank " << rank
          << " send failed, rc " << rc << endl;
      MPI_Finalize();
      exit(1);
    }
  }

  }
  memcpy((char *)&before2D_inverse[myRows_len * rank], (char *)&after1D_inverse[myRows_len * rank], myRows_len * sizeof(Complex));

      // Every CPU receive 1D_inverse
  Transpose(before2D_inverse, width, height, before2D_inverse_Transpose) ;

  Transform1D_myRows_Inverse(before2D_inverse_Transpose, width, startRow, myRows, after2D_inverse) ;
    memcpy((char *)&before1D_inverse[myRows_len * rank], (char *)&after2D[myRows_len * rank], myRows_len * sizeof(Complex));


  rc = MPI_Isend(&after2D_inverse[myRows_len * rank], myRows_len * sizeof(Complex) , MPI_CHAR, 0, 0, MPI_COMM_WORLD, &request);
  if (rc != MPI_SUCCESS)
  {
    cout << "Rank " << rank
        << " send failed, rc " << rc << endl;
    MPI_Finalize();
    exit(1);
  }

    if (rank == 0) {
    for (int cpuN = 1; cpuN < nCPUs; ++cpuN) {
      MPI_Status status;
      rc = MPI_Recv(&collect2D_inverse[myRows_len * cpuN],myRows_len * sizeof(Complex), MPI_CHAR, cpuN , 0, MPI_COMM_WORLD, &status);
      if (rc != MPI_SUCCESS)
      {
        cout << "Rank " << rank
            << " send failed, rc " << rc << endl;
        MPI_Finalize();
        exit(1);
      }
   
    }
    memcpy((char *)&collect2D_inverse[myRows_len * rank], (char *)&after2D_inverse[myRows_len * rank], myRows_len * sizeof(Complex));


    image.SaveImageData("MyAfter2dInverse.txt",collect2D_inverse,width,height);

  }

  MPI_Barrier(MPI_COMM_WORLD);

}



void Transpose(Complex* input, int width, int height, Complex* output)
{
  for(int i = 0; i < width; i++)
  {
    for(int j = 0; j < height; j++)
    {
      output[i*width+j] = input[i+j*width];
    }
  }
}

void Transform1D_row(Complex* input, int width, Complex* output){     
  for (int i = 0; i < width ; i++){
    for (int j = 0; j< width ; j++){
        Complex temp ;
        double Wnk = 2 * M_PI *i*j / width ;
        temp.real = cos(Wnk) ;
        temp.imag = -sin(Wnk) ;
        output[i] = output[i]+ temp * input[j] ;
    }
  }
}

void Transform1D_myRows(Complex* in, int width, int myRows_start, int myRows, Complex* out){
  for(int r = myRows_start ; r < myRows_start + myRows ; r++)
  {
    Transform1D_row(&in[width*r], width, &out[width*r]) ;
  }
}

void Transform1D_row_Inverse(Complex* input, int width, Complex* output){  
  Complex factor ;
  factor.real = width ;
  factor.imag = 0 ;   
  for (int i = 0; i < width ; i++){
    for (int j = 0; j< width ; j++){
        Complex temp ;
        double Wnk = 2 * M_PI *i*j / width ;
        temp.real = cos(Wnk) ;
        temp.imag = sin(Wnk) ;
        output[i] = output[i]+ temp * input[j] ;
    }
    output[i] = output[i] / factor ;
  }
}

void Transform1D_myRows_Inverse(Complex* in, int width, int myRows_start, int myRows, Complex* out){
  for(int r = myRows_start ; r < myRows_start + myRows ; r++)
  {
    Transform1D_row_Inverse(&in[width*r], width, &out[width*r]) ;
  }
}

int main(int argc, char** argv)
{
  rc = MPI_Init(&argc,&argv);
  if (rc != MPI_SUCCESS) {
    printf ("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  // MPI initialization here
  Transform2D(fn.c_str()); // Perform the transform.
  // Finalize MPI here
   MPI_Finalize();
}  


  
