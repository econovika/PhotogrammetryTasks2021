#include "triangulation.h"

#include "defines.h"
#include <iostream>
#include <eigen3/Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    // составление однородной системы + SVD
    // без подвохов
    if (count != 2)
        std::runtime_error("Invalid count value");

    double x0 = ms[0][0];
    double y0 = ms[0][1];
    double x1 = ms[1][0];
    double y1 = ms[1][1];

    matrix4d Acv;
    cv::Matx<double, 1, 4> vec0 = x0 * Ps[0].row(2) - Ps[0].row(0);
    cv::Matx<double, 1, 4> vec1 = y0 * Ps[0].row(2) - Ps[0].row(1);
    cv::Matx<double, 1, 4> vec2 = x1 * Ps[1].row(2) - Ps[1].row(0);
    cv::Matx<double, 1, 4> vec3 = y1 * Ps[1].row(2) - Ps[1].row(1);

    Acv << vec0(0, 0), vec0(0, 1), vec0(0, 2), vec0(0, 3),
           vec1(0, 0), vec1(0, 1), vec1(0, 2), vec1(0, 3),
           vec2(0, 0), vec2(0, 1), vec2(0, 2), vec2(0, 3),
           vec3(0, 0), vec3(0, 1), vec3(0, 2), vec3(0, 3);

    Eigen::Matrix4d A;
    copy(Acv, A);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd vec = svd.matrixV().col(3);

    cv::Vec4d result;
    result << vec[0], vec[1], vec[2], vec[3];
    return result;
}
