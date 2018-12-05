#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
#define rep(i, n) for (int i = 0; i < n; i++)
#include <ctime>

Mat getEnergy(Mat img) {
    Mat gray, sobelX, sobelY;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Sobel(gray, sobelX, CV_64F, 1, 0);
    Sobel(gray, sobelY, CV_64F, 0, 1);
    //print(abs(sobelX) + abs(sobelY));
    return abs(sobelX) + abs(sobelY);
}

Mat getSeam(Mat energy, double *E = nullptr) {
    int rows = energy.rows, cols = energy.cols;
    Mat f(rows, cols, CV_64F), prej(rows, cols, CV_32S);
    //auto start = clock();
    rep(i, rows) rep(j, cols) {
        if (i >= 1) {
            f.at<double>(i, j) = f.at<double>(i - 1, j);
            prej.at<int>(i, j) = j;
            if (j >= 1 && f.at<double>(i - 1, j - 1) < f.at<double>(i, j)) {
                f.at<double>(i, j) = f.at<double>(i - 1, j - 1);
                prej.at<int>(i, j) = j - 1;
            }
            if (j <= cols - 2 && f.at<double>(i - 1, j + 1) < f.at<double>(i, j)) {
                f.at<double>(i, j) = f.at<double>(i - 1, j + 1);
                prej.at<int>(i, j) = j + 1;
            }
        }
        f.at<double>(i, j) += energy.at<double>(i, j);
    }
    //print(f);
    //cout << clock() - start;
    Mat seam(rows, 1, CV_32S);
    double minE = f.at<double>(rows - 1, 0);
    seam.at<int>(rows - 1, 0) = 0;
    rep(i, cols) if (f.at<double>(rows - 1, i) < minE) {
        minE = f.at<double>(rows - 1, i);
        seam.at<int>(rows - 1, 0) = i;
    }
    if (E != nullptr) *E = minE;

    for (int i = rows - 2; i >= 0; i--)
        seam.at<int>(i, 0) = prej.at<int>(i + 1, seam.at<int>(i + 1, 0));

    return seam;
}

Mat drawSeamsV(Mat img, Mat seams) {
    int rows = seams.rows, cols = img.cols, nSeams = seams.cols;
    rep(i, nSeams) {
        Vec3b color(rand() % 256, rand() % 256, rand() % 256);
        rep(j, rows) img.at<Vec3b>(j, seams.at<int>(j, i)) = color;
    }
    return img;
}

Mat drawSeamsH(Mat img, Mat seams) {
    int rows = img.rows, cols = seams.rows, nSeams = seams.cols;
    rep(i, nSeams) {
        Vec3b color(rand() % 256, rand() % 256, rand() % 256);
        rep(j, cols) img.at<Vec3b>(seams.at<int>(j, i), j) = color;
    }
    return img;
}

Mat trim(Mat img, Mat seam) {
    int rows = img.rows, cols = img.cols;
    Mat imgNew(rows, cols - 1, CV_8UC3);
    rep(i, rows) rep(j, cols)
        if (j < seam.at<int>(i, 0))
            imgNew.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);
        else if (j > seam.at<int>(i, 0))
            imgNew.at<Vec3b>(i, j - 1) = img.at<Vec3b>(i, j);
    return imgNew;
}

pair<Mat, Mat> getMultiSeams(Mat img, int n) {
    int rows = img.rows, cols = img.cols;
    Mat seams(rows, n, CV_32S);
    rep(i, n) {
        cout << i << endl;
        auto energy = getEnergy(img);
        auto seam = getSeam(energy);
        img = trim(img, seam);
        int offset = 0;
        rep(j, i) if (seams.at<int>(rows - 1, j) <= seam.at<int>(rows - 1, 0)) offset++;
        rep(j, rows) seams.at<int>(j, i) = seam.at<int>(j, 0) + offset;
        //imwrite("y.bmp", img);
    }
    return make_pair(seams, img);
}

pair<Mat, pair<Mat, Mat> > shrink(Mat img, double p = 0.2) {
    int rows = img.rows, cols = img.cols, nV = int(cols * p), nH = int(rows * p);
    auto tmp = getMultiSeams(img, nV);
    img = tmp.second;
    auto seamsV = tmp.first;
    img = img.t();
    tmp = getMultiSeams(img, nH);
    img = tmp.second;
    auto seamsH = tmp.first;
    img = img.t();

    vector<pair<int, int> > candi;
    rep(i, nV) candi.emplace_back(make_pair(seamsV.at<int>(rows - 1, i), i));
    sort(candi.begin(), candi.end());
    //print(seamsH);

    rep(i, nV) {
        Mat newSeamsH(seamsH.rows + 1, nH, CV_32S);
        rep(j, nH) {
            int ii = candi[i].second;
            int pos;
            //print(seamsV.col(ii));
            //if (j == 0 && i > 100)
            //    print(seamsH.col(j));
            /*if (i == 1 && j == 1) {
                cout << "!!!";
            }*/
            int minDis = 999;

            rep(pp, seamsH.rows) {
                //cout << pos << " " << seamsH.at<int>(pos, j) << endl;
                //if (i == 113 && j == 0) cout << abs(pos - seamsV.at<int>(seamsH.at<int>(pos, j), ii)) << endl;
                int tmp = abs(pp - seamsV.at<int>(seamsH.at<int>(pp, j), ii));
                if (tmp < minDis) {
                    minDis = tmp;
                    pos = pp;
                }
            }
            rep(k, pos + 1) newSeamsH.at<int>(k, j) = seamsH.at<int>(k, j);
            //if (pos + 1 == newSeamsH.rows) newSeamsH.at<int>(pos, j) = seamsH.at<int>(pos - 1, j);
            //else newSeamsH.at<int>(pos, j) = seamsH.at<int>(pos, j);
            for (int k = pos + 1; k < newSeamsH.rows; k++) newSeamsH.at<int>(k, j) = seamsH.at<int>(k - 1, j);
        }
        seamsH = newSeamsH;
        //print(seamsH);
    }

    return make_pair(img, make_pair(seamsV, seamsH));
}

void task1() {
    string root = "Images/";
    string files[]{"3.jpg", "2.png", "1.jpg", "4.jpg", "5.jpg", "6.jpg", "cube.png"};
    rep(i, 6) {
        auto img = imread(root + files[i]);
        auto tmp = shrink(img);
        auto seamsV = tmp.second.first, seamsH = tmp.second.second;
        auto imgDraw = drawSeamsV(img, seamsV);
        imgDraw = drawSeamsH(img, seamsH);
        string rootOut = "ImagesShrunk/";
        imwrite(rootOut + "seams_" + files[i], imgDraw);
        auto imgNew = tmp.first;
        imwrite(rootOut + "shunk_" + files[i], imgNew);
    }
}

int main() {
    task1();
}