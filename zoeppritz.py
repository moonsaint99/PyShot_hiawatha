import numpy as np

def ref_trans_array(mat1, mat2, theta1):
    theta2 = np.arcsin(mat2[1]/mat1[1] * np.sin(theta1))
    phi1 = np.arcsin(mat1[2]/mat1[1] * np.sin(theta1))
    phi2 = np.arcsin(mat2[2]/mat1[1] * np.sin(theta1))
    matrix = np.array([[-np.sin(theta1), -np.cos(phi1), np.sin(theta2), np.cos(phi2)],
                       [np.cos(theta1), -np.sin(phi1), np.cos(theta2), -np.sin(phi2)],
                       [np.sin(2*theta1), mat1[1]/mat1[2]*np.cos(2*phi1),
                        mat2[0]*mat2[2]**2*mat1[1]/(mat1[0]*mat1[2]**2*mat2[1])*np.sin(2*theta2),
                        mat2[0]*mat2[2]*mat1[1]/(mat1[0]*mat1[2]**2)*np.cos(2*phi2)],
                       [-np.cos(2*phi1), mat1[2]/mat1[1]*np.sin(2*phi1),
                        mat2[0]*mat2[1]/(mat1[0]*mat1[1])*np.cos(2*phi2),
                        -mat2[0]*mat2[2]/(mat1[0]*mat1[1])*np.sin(2*phi2)]])
    matrix = np.linalg.inv(matrix)
    array = np.array([np.sin(theta1), np.cos(theta1), np.sin(2*theta1), np.cos(2*phi1)])
    return np.matmul(matrix, array)