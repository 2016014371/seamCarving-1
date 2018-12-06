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
    int rows = img.rows, cols = img.cols;
    rep(i, rows) {
        sobelX.at<double>(i, 0) = sobelX.at<double>(i, 1);
        sobelX.at<double>(i, cols - 1) = sobelX.at<double>(i, cols - 2);
    }
    return abs(sobelX);// + abs(sobelY);
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
    rep(i, rows) rep(j, cols)if (j < seam.at<int>(i, 0))
                imgNew.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);
            else if (j > seam.at<int>(i, 0))
                imgNew.at<Vec3b>(i, j - 1) = img.at<Vec3b>(i, j);
    return imgNew;
}

Mat trimMask(Mat img, Mat seam) {
    int rows = img.rows, cols = img.cols;
    Mat imgNew(rows, cols - 1, CV_64F);
    rep(i, rows) rep(j, cols)if (j < seam.at<int>(i, 0))
                imgNew.at<double>(i, j) = img.at<double>(i, j);
            else if (j > seam.at<int>(i, 0))
                imgNew.at<double>(i, j - 1) = img.at<double>(i, j);
    return imgNew;
}

pair <Mat, Mat> getMultiSeams(Mat img, int n, Mat *mask = nullptr) {
    int rows = img.rows, cols = img.cols;
    Mat seams(rows, n, CV_32S);
    rep(i, n) {
        cout << i << endl;
        auto energy = getEnergy(img);
        if (mask != nullptr) {
            rep(r, energy.rows) rep(c, energy.cols) energy.at<double>(r, c) += mask->at<double>(r, c);
            //print(energy);
        }
        auto seam = getSeam(energy);
        img = trim(img, seam);
        if (mask != nullptr) {
            *mask = trimMask(*mask, seam);
        }
        int offset = 0;
        rep(j, i) if (seams.at<int>(rows - 1, j) <= seam.at<int>(rows - 1, 0)) offset++;
        rep(j, rows) seams.at<int>(j, i) = seam.at<int>(j, 0) + offset;
        //imwrite("y.bmp", img);
    }
    return make_pair(seams, img);
}


pair <Mat, Mat> getSeamsToEnd(Mat img, Mat *mask) {
    int rows = img.rows, cols = img.cols;
    Mat seams(rows, 0, CV_32S);
    int i = 0;
    for(;;) {
        cout << i << endl;
        double E;
        auto energy = getEnergy(img);

        rep(r, energy.rows) rep(c, energy.cols) energy.at<double>(r, c) += mask->at<double>(r, c);

        auto seam = getSeam(energy, &E);
        if (E > 0) break;

        img = trim(img, seam);

        *mask = trimMask(*mask, seam);
        hconcat(seams, seam, seams);
        int offset = 0;
        rep(j, i) if (seams.at<int>(rows - 1, j) <= seam.at<int>(rows - 1, 0)) offset++;
        rep(j, rows) seams.at<int>(j, i) = seam.at<int>(j, 0) + offset;
        //imwrite("y.bmp", img);
        i++;
    }
    return make_pair(seams, img);
}

pair <Mat, pair<Mat, Mat>> shrink(Mat img, Mat *mask = nullptr, double p = 0.2) {
    int rows = img.rows, cols = img.cols, nV = int(cols * p), nH = int(rows * p);
    auto tmp = getMultiSeams(img, nV, mask);
    img = tmp.second;
    auto seamsV = tmp.first;
    img = img.t();
    //rep(r, mask->rows) rep(c, mask->cols) mask->at<double>(r, c) =
    *mask = mask->t();
    tmp = getMultiSeams(img, nH, mask);
    img = tmp.second;
    auto seamsH = tmp.first;
    img = img.t();

    vector <pair<int, int>> candi;
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
    string files[]{"5.jpg", "3.jpg", "2.png", "1.jpg", "4.jpg", "6.jpg", "cube.png"};
    rep(i, 7) {
        auto img = imread(root + files[i]);
        auto tmp = shrink(img);
        auto seamsV = tmp.second.first, seamsH = tmp.second.second;
        auto imgDraw = drawSeamsV(img, seamsV);
        imgDraw = drawSeamsH(imgDraw, seamsH);
        string rootOut = "ImagesShrunk/";
        imwrite(rootOut + "seams_" + files[i], imgDraw);
        auto imgNew = tmp.first;
        imwrite(rootOut + "shrunk_" + files[i], imgNew);
    }
}


void taskAmp() {
    string root = "ImagesToAmp/";
    string rootOut = "ImagesAmp/";
    string files[]{"2.png"};
    rep(i, 1) {
        auto img = imread(root + files[i]);
        Mat imgAmp;
        resize(img, imgAmp, Size(int(img.cols * 1.25), int(img.rows * 1.25)));
        auto tmp = shrink(imgAmp);
        auto seamsV = tmp.second.first, seamsH = tmp.second.second;
        imwrite(rootOut + "resize_" + files[i], imgAmp);
        auto imgDraw = drawSeamsV(imgAmp, seamsV);
        imgDraw = drawSeamsH(imgDraw, seamsH);

        imwrite(rootOut + "seams_" + files[i], imgDraw);
        auto imgNew = tmp.first;
        imwrite(rootOut + "amp_" + files[i], imgNew);
    }
}

bool isWhite(Vec3b c) {
    return c[0] > 250 && c[1] > 250 && c[2] > 250;
}

void taskProtect() {
    string root = "ImagesToPro/";
    string rootOut = "ImagesPro/";
    string files[]{"cube.png"};
    rep(i, 1) {
        auto img = imread(root + files[i]);
        auto ref = imread(root + "mask_" + files[i]);
        Mat mask0(img.rows, img.cols, CV_64F);
        rep(r, img.rows) rep(c, img.cols) {
                //print(ref.at<Vec3b>(r, c));
                if (!isWhite(ref.at<Vec3b>(r, c)))
                    mask0.at<double>(r, c) = 999999;
            }
        //print(mask0);
        Mat *mask = &mask0;
        //imwrite(rootOut + "mask_" + files[i], mask0);
        auto tmp = shrink(img, mask);
        auto seamsV = tmp.second.first, seamsH = tmp.second.second;
        auto imgDraw = drawSeamsV(img, seamsV);
        imgDraw = drawSeamsH(imgDraw, seamsH);

        imwrite(rootOut + "seams_" + files[i], imgDraw);
        auto imgNew = tmp.first;
        imwrite(rootOut + "shrunk_" + files[i], imgNew);
    }
}

void taskRemove() {
    string root = "ImagesToRemove/";
    string rootOut = "ImagesRemoved/";
    string files[]{"1.jpg"};
    rep(i, 1) {
        auto img = imread(root + files[i]);
        auto ref = imread(root + "mask_" + files[i]);
        Mat mask0(img.rows, img.cols, CV_64F);
        rep(r, img.rows) rep(c, img.cols) {
                //print(ref.at<Vec3b>(r, c));
                if (!isWhite(ref.at<Vec3b>(r, c)))
                    mask0.at<double>(r, c) = -999999;
            }
        Mat *mask = &mask0;
        auto tmp = getSeamsToEnd(img, mask);
        auto seamsV = tmp.first;
        auto imgDraw = drawSeamsV(img, seamsV);

        imwrite(rootOut + "seams_" + files[i], imgDraw);
        auto imgNew = tmp.second;
        imwrite(rootOut + "removed_" + files[i], imgNew);
    }
}

void taskEnlarge() {
    string root = "ImagesToEnlarge/";
    string rootOut = "ImagesEnlarged/";
    string files[]{"4.jpg"};
    rep(i, 1) {
        auto img = imread(root + files[i]);
        int rows = img.rows, cols = img.cols, n = int(cols * 0.2);
        auto tmp = getMultiSeams(img, n);

        auto seamsV = tmp.first;

        vector <pair<int, int>> candi;
        rep(i, n) candi.emplace_back(make_pair(seamsV.at<int>(rows - 1, i), i));
        sort(candi.begin(), candi.end());

        Mat imgNew(rows, cols + n, CV_8UC3);
        rep(r, rows) rep(c, cols) imgNew.at<Vec3b>(r, c) = img.at<Vec3b>(r, c);

        imwrite("x.bmp", imgNew);

        rep(ii, n) {
            int iii = candi[ii].second;
            rep(r, rows) {
                int pos = seamsV.at<int>(r, iii);
                for (int c = imgNew.cols - 1; c >= pos + 1; c--)
                    imgNew.at<Vec3b>(r, c) = imgNew.at<Vec3b>(r, c - 1);
                imgNew.at<Vec3b>(r, pos) = Vec3b((Vec3f(imgNew.at<Vec3b>(r, pos)) + Vec3f(imgNew.at<Vec3b>(r, pos - 1))) / 2);
            }
            //imwrite(to_string(ii) + "x.bmp", imgNew);
            for (int j = ii + 1; j < n; j++) rep(r, rows) seamsV.at<int>(r, candi[j].second)++;
        }

        imwrite(rootOut + "enlarged_" + files[i], imgNew);

        auto imgDraw = drawSeamsV(imgNew, seamsV);

        imwrite(rootOut + "seams_" + files[i], imgDraw);

    }
}


int main() {
    taskEnlarge();
}