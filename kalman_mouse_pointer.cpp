#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

cv::Point mousePos(-1,-1);

void mouseCallback(int event, int x, int y, int flags, void* userdata){
    if(event == cv::EVENT_MOUSEMOVE){
        mousePos = cv::Point(x,y);
    }
}

struct PointGroup {
    cv::Point mouse;
    cv::Point predict;
    cv::Point estimate;
};

int main(){
    cv::KalmanFilter kf(4,2,0);
    kf.transitionMatrix = (cv::Mat_<float>(4,4) <<
        1,0,1,0,
        0,1,0,1,
        0,0,1,0,
        0,0,0,1    
    );
    kf.measurementMatrix = (cv::Mat_<float>(2,4) <<
        1,0,0,0,
        0,1,0,0
    );
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-4));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));

    cv::Mat img(600,800, CV_8UC3, cv::Scalar(0,0,0));
    cv::namedWindow("mouse tracking");
    cv::setMouseCallback("mouse tracking", mouseCallback);

    kf.statePost.at<float>(0) = 0;
    kf.statePost.at<float>(1) = 0;

    std::vector<PointGroup> history;

    while(true){
        cv::Mat prediction = kf.predict();
        cv::Point predictPt(prediction.at<float>(0), prediction.at<float>(1));

        cv::Point estimatedPt = predictPt;
        if(mousePos.x >= 0 && mousePos.y >= 0){
            cv::Mat measurement = (cv::Mat_<float>(2,1) << (float)mousePos.x, (float)mousePos.y);
            cv::Mat estimated = kf.correct(measurement);
            estimatedPt = cv::Point(estimated.at<float>(0), estimated.at<float>(1));
        }

        history.push_back({mousePos, predictPt, estimatedPt});
        if(history.size() > 166){
            history.erase(history.begin());
        }

        img = cv::Scalar(0,0,0);
        
        for(size_t i = 0; i < history.size(); ++i){
            float alpha = (float)i / history.size();
            
            if(history[i].mouse.x >= 0){
                cv::circle(img, history[i].mouse, 4, cv::Scalar(0, 0, 255 * alpha), -1);
            }
            cv::circle(img, history[i].predict, 4, cv::Scalar(0, 255 * alpha, 255 * alpha), -1);
            cv::circle(img, history[i].estimate, 4, cv::Scalar(0, 255 * alpha, 0), -1);
        }

        cv::imshow("mouse tracking", img);
        if(cv::waitKey(30) == 27) break;
    }
    return 0;
}
