To begin to understand the work-sharing constructs, we need to learn how to parallelise the `for` or `do` loop.
For this, we will learn simple vector addition examples.

<figure markdown>
![](../figures/vector_add-external.png){ align=middle}
<figcaption></figcaption>
</figure>


As we can see from the above figure, the two vectors should be added to get a single vector. This is done by iterating over the elements and adding them together. For this, we use `for` or `do`. Since there are no data dependencies, the loop indexes do not have any data dependency on the other indexes. Therefore, it is easy to parallelise.



??? example "Examples: Loops"

    === "Serial(C/C++)"

        ```c  
        for(int i = 0; i < n; i ++)
          {
            c[i] = a[i] + b[i];
          }
        ```  
        

    === "FORTRAN(C/C++)"
    
        ```c  
        do i = 1, n
          c(i) = a(i) + b(i)
        end do
        ```  


### <u>Questions and Solutions</u>


??? example "Examples: Vector Addition"


    === "Serial(C/C++)"
    
        ```c  
        //-*-C++-*-
        // Vector-addition.c
        
        #include <stdio.h>
        #include <stdlib.h>
        #include <math.h>
        #include <assert.h>
        #include <time.h>
        
        #define N 5120
        #define MAX_ERR 1e-6

        // CPU function that adds two vector 
        float * Vector_Add(float *a, float *b, float *out, int n) 
        {
          for(int i = 0; i < n; i ++)
            {
              out[i] = a[i] + b[i];
            }
          return out;
        }

        int main()
        {
          // Initialize the variables
          float *a, *b, *out;       
  
          // Allocate the memory
          a   = (float*)malloc(sizeof(float) * N);
          b   = (float*)malloc(sizeof(float) * N);
          out = (float*)malloc(sizeof(float) * N);
  
          // Initialize the arrays
          for(int i = 0; i < N; i++)
            {
              a[i] = 1.0f;
              b[i] = 2.0f;
            }
    
          // Start measuring time
          clock_t start = clock();

          // Executing vector addtion function 
          Vector_Add(a, b, out, N);

          // Stop measuring time and calculate the elapsed time
          clock_t end = clock();
          double elapsed = (double)(end - start)/CLOCKS_PER_SEC;
        
          printf("Time measured: %.3f seconds.\n", elapsed);
  
          // Verification
          for(int i = 0; i < N; i++)
            {
              assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
            }

          printf("out[0] = %f\n", out[0]);
          printf("PASSED\n");
    
          // Deallocate the memory
          free(a); 
          free(b); 
          free(out);
   
          return 0;
        }
        ```

    === "Serial(FORTRAN)"
        ```c
        module Vector_Addition_Mod  
        implicit none 
          contains
        subroutine Vector_Addition(a, b, c, n)
        ! Input vectors
        real(8), intent(in), dimension(:) :: a
        real(8), intent(in), dimension(:) :: b
        real(8), intent(out), dimension(:) :: c
        integer :: i, n
          do i = 1, n
            c(i) = a(i) + b(i)
          end do
         end subroutine Vector_Addition
        end module Vector_Addition_Mod

        program main
        use Vector_Addition_Mod
        implicit none
        ! Input vectors
        real(8), dimension(:), allocatable :: a
        real(8), dimension(:), allocatable :: b 
        ! Output vector
        real(8), dimension(:), allocatable :: c
        ! real(8) :: sum = 0

        integer :: n, i  
        print *, "This program does the addition of two vectors "
        print *, "Please specify the vector size = "
        read *, n

        ! Allocate memory for vector
        allocate(a(n))
        allocate(b(n))
        allocate(c(n))
  
        ! Initialize content of input vectors, 
        ! vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
        do i = 1, n
          a(i) = sin(i*1D0) * sin(i*1D0)
          b(i) = cos(i*1D0) * cos(i*1D0) 
        enddo
    
        ! Call the vector add subroutine 
        call Vector_Addition(a, b, c, n)

        !!Verification
        do i = 1, n
          if (abs(c(i)-(a(i)+b(i)) == 0.00000)) then 
           else
             print *, "FAIL"
           endif
        enddo
        print *, "PASS"
    
        ! Delete the memory
        deallocate(a)
        deallocate(b)
        deallocate(c)
  
        end program main

        ```



    === "Template(C/C++)"
    
        ```c
        //-*-C++-*-
        // Vector-addition.c
        
        #include <stdio.h>
        #include <stdlib.h>
        #include <math.h>
        #include <assert.h>
        #include <time.h>
        
        #define N 5120
        #define MAX_ERR 1e-6

        // CPU function that adds two vector 
        float * Vector_Add(float *a, float *b, float *out, int n) 
        {
        // ADD YOUR PARALLEL REGION FOR THE LOOP
          for(int i = 0; i < n; i ++)
            {
              out[i] = a[i] + b[i];
            }
          return out;
        }

        int main()
        {
          // Initialize the variables
          float *a, *b, *out;       
  
          // Allocate the memory
          a   = (float*)malloc(sizeof(float) * N);
          b   = (float*)malloc(sizeof(float) * N);
          out = (float*)malloc(sizeof(float) * N);
  
          // Initialize the arrays
          for(int i = 0; i < N; i++)
            {
              a[i] = 1.0f;
              b[i] = 2.0f;
            }
    
          // Start measuring time
          clock_t start = clock();

          // ADD YOUR PARALLEL REGION HERE	
          // Executing vector addtion function 
          Vector_Add(a, b, out, N);

          // Stop measuring time and calculate the elapsed time
          clock_t end = clock();
          double elapsed = (double)(end - start)/CLOCKS_PER_SEC;
        
          printf("Time measured: %.3f seconds.\n", elapsed);
  
          // Verification
          for(int i = 0; i < N; i++)
            {
              assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
            }

          printf("out[0] = %f\n", out[0]);
          printf("PASSED\n");
    
          // Deallocate the memory
          free(a); 
          free(b); 
          free(out);
   
          return 0;
        }

        ```
        
    === "Template(FORTRAN)"
        ```c
        module Vector_Addition_Mod  
        implicit none 
          contains
        subroutine Vector_Addition(a, b, c, n)
        use omp_lib
        ! Input vectors
        real(8), intent(in), dimension(:) :: a
        real(8), intent(in), dimension(:) :: b
        real(8), intent(out), dimension(:) :: c
        integer :: i, n
        !! ADD YOUR PARALLEL DO LOOP
          do i = 1, n
            c(i) = a(i) + b(i)
          end do
         end subroutine Vector_Addition
        end module Vector_Addition_Mod

        program main
        use Vector_Addition_Mod
        implicit none
        ! Input vectors
        real(8), dimension(:), allocatable :: a
        real(8), dimension(:), allocatable :: b 
        ! Output vector
        real(8), dimension(:), allocatable :: c
        ! real(8) :: sum = 0

        integer :: n, i  
        print *, "This program does the addition of two vectors "
        print *, "Please specify the vector size = "
        read *, n

        ! Allocate memory for vector
        allocate(a(n))
        allocate(b(n))
        allocate(c(n))
  
        ! Initialize content of input vectors, 
        ! vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
        do i = 1, n
          a(i) = sin(i*1D0) * sin(i*1D0)
          b(i) = cos(i*1D0) * cos(i*1D0) 
        enddo

        !! ADD YOUR PARALLEL REGION 
        ! Call the vector add subroutine 
        call Vector_Addition(a, b, c, n)

        !!Verification
        do i = 1, n
          if (abs(c(i)-(a(i)+b(i)) == 0.00000)) then 
           else
             print *, "FAIL"
           endif
        enddo
        print *, "PASS"
    
        ! Delete the memory
        deallocate(a)
        deallocate(b)
        deallocate(c)
  
        end program main

        ```

    === "Solution(C/C++)"
    
        ```c
        //-*-C++-*-
        // Vector-addition.c
        
        #include <stdio.h>
        #include <stdlib.h>
        #include <math.h>
        #include <assert.h>
        #include <time.h>
        
        #define N 5120
        #define MAX_ERR 1e-6

        // CPU function that adds two vector 
        float * Vector_Add(float *a, float *b, float *out, int n) 
        #pragma omp for
        // ADD YOUR PARALLEL REGION FOR THE LOOP
          for(int i = 0; i < n; i ++)
            {
              out[i] = a[i] + b[i];
            }
          return out;
        }

        int main()
        {
          // Initialize the variables
          float *a, *b, *out;       
  
          // Allocate the memory
          a   = (float*)malloc(sizeof(float) * N);
          b   = (float*)malloc(sizeof(float) * N);
          out = (float*)malloc(sizeof(float) * N);
  
          // Initialize the arrays
          for(int i = 0; i < N; i++)
            {
              a[i] = 1.0f;
              b[i] = 2.0f;
            }
    
          // Start measuring time
          clock_t start = clock();

          #pragma omp parallel 
          // Executing vector addtion function 
          Vector_Add(a, b, out, N);

          // Stop measuring time and calculate the elapsed time
          clock_t end = clock();
          double elapsed = (double)(end - start)/CLOCKS_PER_SEC;
        
          printf("Time measured: %.3f seconds.\n", elapsed);
  
          // Verification
          for(int i = 0; i < N; i++)
            {
              assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
            }

          printf("out[0] = %f\n", out[0]);
          printf("PASSED\n");
    
          // Deallocate the memory
          free(a); 
          free(b); 
          free(out);
   
          return 0;
        }

        ```

    === "Solution(FORTRAN)"
        ```c
        module Vector_Addition_Mod  
        implicit none 
          contains
        subroutine Vector_Addition(a, b, c, n)
        use omp_lib
        ! Input vectors
        real(8), intent(in), dimension(:) :: a
        real(8), intent(in), dimension(:) :: b
        real(8), intent(out), dimension(:) :: c
        integer :: i, n
        !$omp do
          do i = 1, n
            c(i) = a(i) + b(i)
          end do
        !$omp end do
         end subroutine Vector_Addition
        end module Vector_Addition_Mod

        program main
        use Vector_Addition_Mod
        implicit none
        ! Input vectors
        real(8), dimension(:), allocatable :: a
        real(8), dimension(:), allocatable :: b 
        ! Output vector
        real(8), dimension(:), allocatable :: c
        ! real(8) :: sum = 0

        integer :: n, i  
        print *, "This program does the addition of two vectors "
        print *, "Please specify the vector size = "
        read *, n

        ! Allocate memory for vector
        allocate(a(n))
        allocate(b(n))
        allocate(c(n))
  
        ! Initialize content of input vectors, 
        ! vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
        do i = 1, n
          a(i) = sin(i*1D0) * sin(i*1D0)
          b(i) = cos(i*1D0) * cos(i*1D0) 
        enddo

        !$omp parallel 
        ! Call the vector add subroutine 
        call Vector_Addition(a, b, c, n)
        !$omp end parallel
        
        !!Verification
        do i = 1, n
          if (abs(c(i)-(a(i)+b(i)) == 0.00000)) then 
           else
             print *, "FAIL"
           endif
        enddo
        print *, "PASS"
    
        ! Delete the memory
        deallocate(a)
        deallocate(b)
        deallocate(c)
  
        end program main

        ```



??? "Compilation and Output"

    === "Serial(C/C++)"
        ```c
        // compilation
        $ gcc Vector-addition-Serial.c -o Vector-addition-Serial
        
        // execution 
        $ ./Vector-addition-Serial
        
        // output
        $ ./Vector-addition-Serial
        ```
        
    === "Serial(FORTRAN)"
        ```c
        // compilation
        $ gfortran Vector-addition-Serial.f90 -o Vector-addition-Serial
        
        // execution
        $ ./Vector-addition-Serial
        
        // output
        $ ./Vector-addition-Serial
        ```


    === "Solution(C/C++)"
        ```c
        // compilation
        $ gcc -fopennmp Vector-addition-OpenMP-solution.c -o Vector-addition-Solution
        
        // execution 
        $ ./Vector-addition-Solution
        
        // output
        $ ./Vector-addition-Solution
        ```
        
    === "Solution(FORTRAN)"
        ```c
        // compilation
        $ gfortran -fopenmp Vector-addition-OpenMP-solution.f90 -o Vector-addition-Solution
        
        // execution
        $ ./Vector-addition-Solution
        
        // output
        $ ./Vector-addition-Solution
        ```


??? Question "Questions"

    - What happens if you remove the **`__syncthreads();`** from the **`__global__ void vector_add(float *a, float *b, 
       float *out, int n)`** function.
    - Can you remove the if condition **`if(i < n)`** from the **`__global__ void vector_add(float *a, float *b,
       float *out, int n)`** function. If so how can you do that?
    - Here we do not use the **`cudaDeviceSynchronize()`** in the main application, can you figure out why we
        do not need to use it. 
    - Can you create a different kinds of threads block for larger number of array?