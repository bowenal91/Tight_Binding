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

double epsilon_g, epsilon_e, J_g, J_e; //Epsilon is potential energy, and J is coupling between adjacent sites; g is ground state, and e is excited state
int numLines, numDihedrals, numEnergies, numJunk;
std::vector<double> params;
std::vector<double> lb;
std::vector<double> ub;
double **dihedral_list, **energy_list;
FILE *dataFile;
char *dihedral_file_name, *energy_file_name, *output_name, *HOMO_file_name, *LUMO_file_name;
MatrixXd Hamiltonian;
MatrixXd Eigenvalues;
MatrixXd Eigenvectors;

void set_Hamiltonian(int i) {
    int j,j2;
    for (j=0;j<numEnergies;j++) {
        j2 = j + numEnergies;
        Hamiltonian(j,j) = epsilon_g;
        Hamiltonian(j2,j2) = epsilon_e;
    }

    for (j=0;j<numEnergies-1;j++) {
        j2 = j + numEnergies;
        Hamiltonian(j,j+1) = -J_g*dihedral_list[i][j];
        Hamiltonian(j+1,j) = -J_g*dihedral_list[i][j];

        Hamiltonian(j2,j2+1) = -J_e*dihedral_list[i][j];
        Hamiltonian(j2+1,j2) = -J_e*dihedral_list[i][j];
    }
}

void diagonalize() {
    SelfAdjointEigenSolver<MatrixXd> solver(Hamiltonian);
    Eigenvalues = solver.eigenvalues().asDiagonal();
    Eigenvectors = solver.eigenvectors();
}


double calc_fitness(const std::vector<double> &c, std::vector<double> &grad, void *data) {
    int i,j,j2;
    epsilon_g = c[0];
    epsilon_e = c[1];
    J_g = c[2];
    J_e = c[3];
    double EV, ref;
    double fit = 0.0;
    for (i=0;i<numLines;i++) {
        set_Hamiltonian(i);
        diagonalize();
        j2 = 0;
        for (j=numEnergies-3;j<numEnergies+3;j++) {
            EV = Eigenvalues(j,j);
            ref = energy_list[i][j2];
            fit += (EV-ref)*(EV-ref);
            j2++;
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
    for (i=0;i<numLines;i++) {
        dihedral_list[i] = new double [numDihedrals];
        energy_list[i] = new double [6];
    }

    char tt[2001];
    dataFile = fopen(dihedral_file_name,"r");
    
    for (i=0;i<numLines;i++) {
        for (j=0;j<numDihedrals;j++) {
            fscanf(dataFile, "%lf", &dihedral_list[i][j]);
        }   
    }

    fclose(dataFile);

    dataFile = fopen(energy_file_name,"r");
    
    for (i=0;i<numLines;i++) {
        for (j=0;j<6;j++) {
            fscanf(dataFile, "%lf", &energy_list[i][j]);
        }   
    }

    fclose(dataFile);


}

void init_matrix() {
    Hamiltonian = MatrixXd::Zero(2*numEnergies,2*numEnergies);
    Eigenvalues = MatrixXd::Zero(2*numEnergies,2*numEnergies);
    Eigenvectors = MatrixXd::Zero(2*numEnergies,2*numEnergies);
}


void print_comparison() {
    int i,j,j2;
    double EV,ref;
    double h1,h2,l1,l2;
    FILE *output = fopen(output_name,"w");
    FILE *output2 = fopen(HOMO_file_name,"w");
    FILE *output3 = fopen(LUMO_file_name,"w");
    for (i=0;i<numLines;i++) {
        set_Hamiltonian(i);
        diagonalize();
        for (j=numEnergies-3;j<numEnergies+3;j++) {
            EV = Eigenvalues(j,j);
            
            fprintf(output, "%f\t",EV);

        }
        fprintf(output,"\n");
        for (j=0;j<numEnergies;j++) {
            h1 = Eigenvectors(j,numEnergies-1);
            h2 = Eigenvectors(j+numEnergies,numEnergies-1);
            l1 = Eigenvectors(j,numEnergies);
            l2 = Eigenvectors(j+numEnergies,numEnergies);

            fprintf(output2,"%f\t",h1*h1+h2*h2);
            fprintf(output3,"%f\t",l1*l1+l2*l2);
            
        }
        fprintf(output2,"\n");
        fprintf(output3,"\n");
    }

    fclose(output);
    fclose(output2);
    fclose(output3);

}

int main(int argc, char **argv) {
    numDihedrals = 5;
    numEnergies = numDihedrals+1;
    numJunk = 0;

    //dihedral_file_name = argv[1];
    //energy_file_name = argv[2];
    output_name = "TB_energies.csv";
    dihedral_file_name = "CNN_input_tb.csv";
    energy_file_name = "CNN_output_energies.csv";
    numEnergies = atoi(argv[1]);
    HOMO_file_name = "TB_HOMO.csv";
    LUMO_file_name = "TB_LUMO.csv";
    numDihedrals = numEnergies-1;
    
    int i,j;
    double fit;
    
    init_data();
    init_matrix();

    opt op(LN_SBPLX,4);
    op.set_min_objective(calc_fitness,NULL);
    op.set_xtol_rel(1e-4);
    params.push_back(-6.0);
    params.push_back(-1.0);
    params.push_back(0.0);
    params.push_back(0.0);

    lb.push_back(-10.0);
    lb.push_back(-10.0);
    lb.push_back(-10.0);
    lb.push_back(-10.0);

    ub.push_back(1.0);
    ub.push_back(1.0);
    ub.push_back(10.0);
    ub.push_back(10.0);
    op.set_lower_bounds(lb);
    op.set_upper_bounds(ub);
    result res = op.optimize(params,fit);
    
    epsilon_g = params[0];
    epsilon_e = params[1];
    J_g = params[2];
    J_e = params[3];

    FILE *param_output = fopen("fitted_TB_params.dat","w");
    fprintf(param_output,"%f\n%f\t%f\n%f\t%f\n",fit,epsilon_g,J_g,epsilon_e,J_e);
    fclose(param_output);
    print_comparison();

    

    
    
    
    return 0;
}
