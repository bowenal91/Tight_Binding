/* This program determines the coupling and potential energy terms for a model tight binding hamiltonian used to describe the energy levels of conjugated polymer */

#include <nlopt.hpp>
#include "time.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <cstdlib>
#include <stdio.h>
#define PI 3.14159265358979323846
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;
using namespace nlopt;

double epsilon, A, B; //Epsilon is potential energy, and J is coupling between adjacent sites; g is ground state, and e is excited state
int numLines, numDihedrals, numEnergies, numJunk, engFileWidth;
std::vector<double> params;
std::vector<double> lb;
std::vector<double> ub;
double **dihedral_list, **energy_list, **dipole_list;
FILE *dataFile;
char *dihedral_file_name, *energy_file_name, *output_name, *Osc_file_name, *Dipole_file_name;
MatrixXd Hamiltonian;
MatrixXd Eigenvalues;
MatrixXd Eigenvectors;

void set_Hamiltonian(int i) {
    int j;
    for (j=0;j<numEnergies;j++) {
        Hamiltonian(j,j) = epsilon;
    }

    for (j=0;j<numEnergies-1;j++) {
        Hamiltonian(j,j+1) = -(A*dihedral_list[i][j] + B*dihedral_list[i][j+numDihedrals]);
        Hamiltonian(j+1,j) = -(A*dihedral_list[i][j] + B*dihedral_list[i][j+numDihedrals]);
    }
}

void diagonalize() {
    SelfAdjointEigenSolver<MatrixXd> solver(Hamiltonian);
    Eigenvalues = solver.eigenvalues().asDiagonal();
    Eigenvectors = solver.eigenvectors();
}


double calc_fitness(const std::vector<double> &c, std::vector<double> &grad, void *data) {
    int i,j,j2;
    epsilon = c[0];
    A = c[1];
    B = c[2];
    double EV, ref;
    double fit = 0.0;
    for (i=0;i<numLines;i++) {
        set_Hamiltonian(i);
        diagonalize();
        for (j=0;j<engFileWidth;j++) {
            EV = Eigenvalues(j,j);
            ref = energy_list[i][j];
            fit += (EV-ref)*(EV-ref);
        }

    }

    fit /= double(numLines);
    printf("%f\n",fit);
    return fit;
}

void init_data() {
    
    int i,j; 
    ifstream aFile(dihedral_file_name);
    numLines = 0;
    string line;
    while(getline(aFile,line)) 
        numLines++;
    aFile.close(); 
    dihedral_list = new double *[numLines];
    energy_list = new double *[numLines];
    dipole_list = new double *[numLines];
    for (i=0;i<numLines;i++) {
        dihedral_list[i] = new double [2*numDihedrals];
        energy_list[i] = new double [engFileWidth];
        dipole_list[i] = new double [3*numEnergies];
    }

    char tt[2001];
    dataFile = fopen(dihedral_file_name,"r");
    
    for (i=0;i<numLines;i++) {
        for (j=0;j<2*numDihedrals;j++) {
            fscanf(dataFile, "%lf", &dihedral_list[i][j]);
        }   
    }

    fclose(dataFile);

    dataFile = fopen(energy_file_name,"r");
    
    for (i=0;i<numLines;i++) {
        for (j=0;j<engFileWidth;j++) {
            fscanf(dataFile, "%lf", &energy_list[i][j]);
        }   
    }

    fclose(dataFile);

    dataFile = fopen(Dipole_file_name,"r");
    for (i=0;i<numLines;i++) {
        for (j=0;j<3*numEnergies;j++) {
            fscanf(dataFile, "%lf", &dipole_list[i][j]);
            //printf("%f\n",dipole_list[i][j]);
        }
    }

    fclose(dataFile);

}

void init_matrix() {
    Hamiltonian = MatrixXd::Zero(numEnergies,numEnergies);
    Eigenvalues = MatrixXd::Zero(numEnergies,numEnergies);
    Eigenvectors = MatrixXd::Zero(numEnergies,numEnergies);
}


void print_comparison() {
    int i,j,j2,k;
    double EV,ref;
    double ux,uy,uz,u2,psi;
    FILE *output = fopen(output_name,"w");
    FILE *output2 = fopen(Osc_file_name,"w");
    for (i=0;i<numLines;i++) {
        set_Hamiltonian(i);
        diagonalize();
        for (j=0;j<engFileWidth;j++) {
            EV = Eigenvalues(j,j);
            
            fprintf(output, "%f\t",EV);

        }
        fprintf(output,"\n");
        for (j=0;j<engFileWidth;j++) {
            ux = 0.0; uy = 0.0; uz = 0.0;
            for (k=0;k<numEnergies;k++) {
                psi = Eigenvectors(k,j);
                ux += psi*dipole_list[i][3*k];
                uy += psi*dipole_list[i][3*k+1];
                uz += psi*dipole_list[i][3*k+2];
            }
            u2 = ux*ux+uy*uy+uz*uz;
            fprintf(output2,"%f\t",u2);


        }
        fprintf(output2,"\n");
    }

    fclose(output);
    fclose(output2);

}

int main(int argc, char **argv) {
    numDihedrals = 5;
    numEnergies = numDihedrals+1;
    numJunk = 0;

    //dihedral_file_name = argv[1];
    //energy_file_name = argv[2];
    output_name = "TB_optical_energies.csv";
    dihedral_file_name = "CNN_input_frenkel.csv";
    energy_file_name = "CNN_output_optical_eng.csv";
    numEnergies = atoi(argv[1]);
    Osc_file_name = "TB_output_optical_osc.csv";
    Dipole_file_name = "Transition_dipoles.csv";
    numDihedrals = numEnergies-1;
    engFileWidth = min(numEnergies,5);
    int i,j;
    double fit;
    
    init_data();
    init_matrix();

    opt op(LN_SBPLX,3);
    op.set_min_objective(calc_fitness,NULL);
    op.set_xtol_rel(1e-4);
    params.push_back(3.0);
    params.push_back(1.0);
    params.push_back(1.0);

    lb.push_back(-10.0);
    lb.push_back(-10.0);
    lb.push_back(-10.0);

    ub.push_back(10.0);
    ub.push_back(10.0);
    ub.push_back(10.0);
    op.set_lower_bounds(lb);
    op.set_upper_bounds(ub);
    result res = op.optimize(params,fit);
    
    epsilon = params[0];
    A = params[1];
    B = params[2];

    FILE *param_output = fopen("fitted_TB_params.dat","w");
    fprintf(param_output,"%f\n%f\n%f\t%f\n",fit,epsilon,A,B);
    fclose(param_output);
    print_comparison();

    

    
    
    
    return 0;
}
