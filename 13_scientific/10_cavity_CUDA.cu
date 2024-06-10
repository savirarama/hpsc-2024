#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

#define NX  (41)
#define NY  (41)
#define NT  (500)
#define NIT (50)
#define DX  (2.0 / (NX - 1))
#define DY  (2.0 / (NY - 1))
#define DT  (0.01)
#define RHO (1.0)
#define NU  (0.02)

__global__ void computeBKernel(double *b, double *u, double *v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 

    if (1 <= i && i < NX - 1 && 1 <= j && j < NY - 1) {
        // Calculate new indices
        int idx = j * NX + i;
        int idx_ip1 = j * NX + (i + 1);
        int idx_im1 = j * NX + (i - 1);
        int idx_jp1 = (j + 1) * NX + i;
        int idx_jm1 = (j - 1) * NX + i;

        // Compute b[j][i]
        double term1 = (u[idx_ip1] - u[idx_im1]) / (2 * DX);
        double term2 = (v[idx_jp1] - v[idx_jm1]) / (2 * DY);

        b[idx] = RHO * (1 / DT * (term1 + term2) - pow(term1, 2) - 2 * term1 * term2 - pow(term2, 2));
    }
}

__global__ void updatePKernel(double *p, double *pn, double *b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;  

    if (1 <= i && i < NX - 1 && 1 <= j && j < NY - 1) {
        double dx2 = DX * DX;
        double dy2 = DY * DY;
        double coeff = 2 * (dx2 + dy2);

        p[j * NX + i] = (dy2 * (pn[j * NX + (i + 1)] + pn[j * NX + (i - 1)]) +
                         dx2 * (pn[(j + 1) * NX + i] + pn[(j - 1) * NX + i]) -
                         b[j * NX + i] * dx2 * dy2) / coeff;
    }
}

__global__ void computeP(double *p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 <= j && j < NY) p[j * NX + (NX - 1)] = p[j * NX + (NX - 2)];
    if (0 <= i && i < NX) p[0 * NX + i] = p[1 * NX + i];
    if (0 <= j && j < NY) p[j * NX + 0] = p[j * NX + 1];
    if (0 <= i && i < NX) p[(NY - 1) * NX + i] = 0.0;
}

__global__
void computeUV(double *u, double *v, double *un, double *vn, double *p) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (1 <= i && i < NX - 1 && 1 <= j && j < NY - 1) {
        u[j * NX + i] = un[j * NX + i]
            - un[j * NX + i] * DT / DX * (un[j * NX + i] - un[j * NX + i - 1])
            - un[j * NX + i] * DT / DY * (un[j * NX + i] - un[(j - 1) * NX + i])
            - DT / (2 * RHO * DX) * (p[j * NX + i + 1] - p[j * NX + i - 1])
            + NU * DT / std::pow(DX, 2) * (un[j * NX + i + 1] - 2 * un[j * NX + i] + un[j * NX + i - 1])
            + NU * DT / std::pow(DY, 2) * (un[(j + 1) * NX + i] - 2 * un[j * NX + i] + un[(j - 1) * NX + i]);

        v[j * NX + i] = vn[j * NX + i]
            - vn[j * NX + i] * DT / DX * (vn[j * NX + i] - vn[j * NX + i - 1])
            - vn[j * NX + i] * DT / DY * (vn[j * NX + i] - vn[(j - 1) * NX + i])
            - DT / (2 * RHO * DY) * (p[(j + 1) * NX + i] - p[(j - 1) * NX + i])
            + NU * DT / std::pow(DX, 2) * (vn[j * NX + i + 1] - 2 * vn[j * NX + i] + vn[j * NX + i - 1])
            + NU * DT / std::pow(DY, 2) * (vn[(j + 1) * NX + i] - 2 * vn[j * NX + i] + vn[(j - 1) * NX + i]);
    }
}

__global__ void computeBoundary(double *u, double *v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (0 <= i && i < NX) u[0*NX+i]        = 0.0;    
    if (0 <= j && j < NY) u[j*NX+0]        = 0.0;  
    if (0 <= j && j < NY) u[j*NX+(NX-1)]   = 0.0;   
    if (0 <= i && i < NX) u[(NY-1)*NX+i]   = 1.0;    

    if (0 <= i && i < NX) v[0*NX+i]        = 0.0;    
    if (0 <= i && i < NX) v[(NY-1)*NX+i]   = 0.0;    
    if (0 <= j && j < NY) v[j*NX+0]        = 0.0;    
    if (0 <= j && j < NY) v[j*NX+(NX-1)]   = 0.0;   
}




int main (){
    double *u;
    double *v;
    double *p;
    double *b;

    cudaMallocManaged(&u, NY*NX*sizeof(double));
    cudaMallocManaged(&v, NY*NX*sizeof(double));
    cudaMallocManaged(&p, NY*NX*sizeof(double));
    cudaMallocManaged(&b, NY*NX*sizeof(double));

    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x, (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    
    std::ofstream ufile("u.dat"), vfile("v.dat"), pfile("p.dat");

    for (int n=0; n<NT; n++) {
        //compute b
        computeBKernel<<<blocksPerGrid, threadsPerBlock>>>(b, u, v);

        for (int it=0; it<NIT; it++) {
            //copy p to pn
            double *pn;
            cudaMallocManaged(&pn, NY*NX*sizeof(double));
            cudaMemcpy(pn, p, NY*NX*sizeof(double), cudaMemcpyHostToHost);
            //update p
            updatePKernel<<<blocksPerGrid, threadsPerBlock>>>(p, pn, b);
            cudaFree(pn);

            computeP<<<blocksPerGrid, threadsPerBlock>>>(p);
        }

        double* un;
        double* vn;
        cudaMallocManaged(&un, NY*NX*sizeof(double));
        cudaMallocManaged(&vn, NY*NX*sizeof(double));
        //Copy u and v to un and vn respectivell=y
        cudaMemcpy(un, u, NY*NX*sizeof(double), cudaMemcpyHostToHost);
        cudaMemcpy(vn, v, NY*NX*sizeof(double), cudaMemcpyHostToHost);

        computeUV<<<blocksPerGrid, threadsPerBlock>>>(u, v, un, vn, p);
        cudaFree(un);
        cudaFree(vn);

        computeBoundary<<<blocksPerGrid, threadsPerBlock>>>(u, v);
    
       
        if (n % 10 == 0) {
            for (int j = 0; j < NY; j++) {
                for (int i = 0; i < NX; i++) {
                    ufile << u[j * NX + i] << " ";
                    vfile << v[j * NX + i] << " ";
                    pfile << p[j * NX + i] << " ";
                }
                ufile << "\n";
                vfile << "\n";
                pfile << "\n";
        
            }
        }
    }
    ufile.close();
    vfile.close();
    pfile.close();

    

    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(b);

    cudaDeviceSynchronize();
    



    return 0;
}