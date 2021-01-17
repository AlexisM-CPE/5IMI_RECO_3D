#include "registration.hpp"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkAffineTransform.h"
#include "itkMeanSquaresImageToImageMetricv4.h"
#include <iostream>
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include <itkCenteredTransformInitializer.h>
#include <itkRegistrationParameterScalesFromPhysicalShift.h>
#include <opencv2/plot.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "itkCorrelationImageToImageMetricv4.h"

cv::Mat display;
cv::Ptr<cv::plot::Plot2d> plot;
cv::Mat xData(1, 1500, CV_64FC1); //1 Row, 100 columns, Double
cv::Mat yData(1, 1500, CV_64FC1);

void CommandIterationUpdate::Execute(itk::Object *caller, const itk::EventObject &event)
{
    Execute((const itk::Object *)caller, event);
}

void CommandIterationUpdate::Execute(const itk::Object *object, const itk::EventObject &event)
{
    auto optimizer = static_cast<OptimizerPointer>(object);
    if (!itk::IterationEvent().CheckEvent(&event))
    {
        return;
    }
    std::cout << optimizer->GetCurrentIteration() << "   ";
    std::cout << optimizer->GetValue() << "   ";
    std::cout << optimizer->GetCurrentPosition();
    yData.at<double>(optimizer->GetCurrentIteration()) = optimizer->GetValue();
    xData.at<double>(optimizer->GetCurrentIteration()) = optimizer->GetCurrentIteration();
    // Print the angle for the trace plot
    vnl_matrix<double> p(2, 2);
    p[0][0] = (double)optimizer->GetCurrentPosition()[0];
    p[0][1] = (double)optimizer->GetCurrentPosition()[1];
    p[1][0] = (double)optimizer->GetCurrentPosition()[2];
    p[1][1] = (double)optimizer->GetCurrentPosition()[3];
    vnl_svd<double> svd(p);
    vnl_matrix<double> r(2, 2);
    r = svd.U() * vnl_transpose(svd.V());
    double angle = std::asin(r[1][0]);
    std::cout << " AffineAngle: " << angle * 180.0 / itk::Math::pi << " SCD : " << optimizer->GetStopConditionDescription() << std::endl;
}

TransformType::Pointer registrate_image(std::string filename1, std::string filename2)
{
    using PixelType = float;

    using FixedImageType = itk::Image<PixelType, Dimension>;
    using MovingImageType = itk::Image<PixelType, Dimension>;

    using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>; //RegularStepGradientDescentOptimizerv4<double>; //itk::LBFGSOptimizerv4;

    using MetricType = itk::MeanSquaresImageToImageMetricv4<FixedImageType, MovingImageType>;
    using RegistrationType = itk::ImageRegistrationMethodv4<FixedImageType, MovingImageType, TransformType>;

    MetricType::Pointer metric = MetricType::New();
    OptimizerType::Pointer optimizer = OptimizerType::New();
    RegistrationType::Pointer registration = RegistrationType::New();

    registration->SetMetric(metric);
    registration->SetOptimizer(optimizer);

    TransformType::Pointer transform = TransformType::New();

    using FixedImageReaderType = itk::ImageFileReader<FixedImageType>;
    using MovingImageReaderType = itk::ImageFileReader<MovingImageType>;
    FixedImageReaderType::Pointer fixedImageReader = FixedImageReaderType::New();
    MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();
    fixedImageReader->SetFileName(filename1);
    movingImageReader->SetFileName(filename2);

    registration->SetFixedImage(fixedImageReader->GetOutput());
    registration->SetMovingImage(movingImageReader->GetOutput());

    using TransformInitializerType = itk::CenteredTransformInitializer<TransformType, FixedImageType, MovingImageType>;

    TransformInitializerType::Pointer initializer = TransformInitializerType::New();
    using VectorType = itk::Vector<double, Dimension>;
    VectorType initTranslation;
    initTranslation[0] = -70.0f; //-20.0f;
    initTranslation[1] = -00.0f; //1.0f;

    std::cout << "test : " << transform->GetTranslation()[0] << " : " << transform->GetTranslation()[0] << std::endl;
    initializer->SetTransform(transform);
    initializer->SetFixedImage(fixedImageReader->GetOutput());
    initializer->SetMovingImage(movingImageReader->GetOutput());
    //initializer->MomentsOn();
    initializer->InitializeTransform();
    transform->SetTranslation(initTranslation);
    registration->SetInitialTransform(transform);
    registration->InPlaceOn();

    double translationScale = 1.0f / 1000.0f; //1.0f/1000
    unsigned int maxNumberOfIterations = 500;
    using OptimizerScalesType = OptimizerType::ScalesType;

    OptimizerScalesType optimizerScales(transform->GetNumberOfParameters());
    optimizerScales[0] = 10.0f;
    optimizerScales[1] = 10.0f;
    optimizerScales[2] = 10.0f;
    optimizerScales[3] = 10.0f; //0.1f; // 1.0
    optimizerScales[4] = translationScale;
    optimizerScales[5] = translationScale;

    optimizer->SetScales(optimizerScales);

    double steplength = 1.0f; //1.0f

    optimizer->SetRelaxationFactor(0.01f); //added
    optimizer->SetLearningRate(steplength);
    optimizer->SetMinimumStepLength(0.001); // 0.0001
    optimizer->SetNumberOfIterations(maxNumberOfIterations);
    /*using ScalesEstimatorType = itk::RegistrationParameterScalesFromPhysicalShift<MetricType>;
    ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
    scalesEstimator->SetMetric(metric);
    scalesEstimator->SetTransformForward(false);
    scalesEstimator->SetSmallParameterVariation(0.1f);

    optimizer->SetGradientConvergenceTolerance(0.0000001f);
    optimizer->SetLineSearchAccuracy(0.5f);
    optimizer->SetDefaultStepLength(10.0f);
    //optimizer->TraceOn();
    optimizer->SetMaximumNumberOfFunctionEvaluations(maxNumberOfIterations);
    optimizer->SetScalesEstimator(scalesEstimator);*/

    CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
    optimizer->AddObserver(itk::IterationEvent(), observer);

    constexpr unsigned int numberOfLevels = 1;

    RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
    shrinkFactorsPerLevel.SetSize(1);
    shrinkFactorsPerLevel[0] = 1.2f;

    RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
    smoothingSigmasPerLevel.SetSize(1);
    smoothingSigmasPerLevel[0] = 2.0f;

    registration->SetNumberOfLevels(numberOfLevels);
    registration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);
    registration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);

    try
    {
        registration->Update();
        std::cout << "Optimizer stop condition: "
                  << registration->GetOptimizer()->GetStopConditionDescription()
                  << std::endl;
    }
    catch (const itk::ExceptionObject &err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
    }

    const TransformType::ParametersType finalParameters = registration->GetOutput()->Get()->GetParameters();

    const double finalRotationCenterX = transform->GetCenter()[0];
    const double finalRotationCenterY = transform->GetCenter()[1];
    const double finalTranslationX = finalParameters[4];
    const double finalTranslationY = finalParameters[5];

    const unsigned int numberOfIterations = optimizer->GetCurrentIteration();
    const double bestValue = optimizer->GetValue();

    std::cout << "Result = " << std::endl;
    std::cout << " Center X      = " << finalRotationCenterX << std::endl;
    std::cout << " Center Y      = " << finalRotationCenterY << std::endl;
    std::cout << " Translation X = " << finalTranslationX << std::endl;
    std::cout << " Translation Y = " << finalTranslationY << std::endl;
    std::cout << " Iterations    = " << numberOfIterations << std::endl;
    std::cout << " Metric value  = " << bestValue << std::endl;

    vnl_matrix<double> p(2, 2);
    p[0][0] = (double)finalParameters[0];
    p[0][1] = (double)finalParameters[1];
    p[1][0] = (double)finalParameters[2];
    p[1][1] = (double)finalParameters[3];
    vnl_svd<double> svd(p);
    vnl_matrix<double> r(2, 2);
    r = svd.U() * vnl_transpose(svd.V());
    double angle = std::asin(r[1][0]);

    const double angleInDegrees = angle * 180.0 / itk::Math::pi;

    std::cout << " Scale 1         = " << svd.W(0) << std::endl;
    std::cout << " Scale 2         = " << svd.W(1) << std::endl;
    std::cout << " Angle (degrees) = " << angleInDegrees << std::endl;

    for (auto a : finalParameters)
    {
        std::cout << (double)a << "-----------" << std::endl;
    }
    using ResampleFilterType = itk::ResampleImageFilter<MovingImageType, FixedImageType>;

    ResampleFilterType::Pointer resampler = ResampleFilterType::New();
    resampler->SetTransform(transform);
    resampler->SetInput(movingImageReader->GetOutput());

    FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();
    resampler->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
    resampler->SetReferenceImage(fixedImage);
    resampler->UseReferenceImageOn();
    resampler->SetOutputOrigin(fixedImage->GetOrigin());
    resampler->SetOutputSpacing(fixedImage->GetSpacing());
    resampler->SetOutputDirection(fixedImage->GetDirection());
    resampler->SetDefaultPixelValue(1);

    using OutputPixelType = unsigned char;

    using OutputImageType = itk::Image<OutputPixelType, Dimension>;

    using CastFilterType = itk::CastImageFilter<FixedImageType, OutputImageType>;

    using WriterType = itk::ImageFileWriter<OutputImageType>;

    WriterType::Pointer writer = WriterType::New();
    CastFilterType::Pointer caster = CastFilterType::New();

    writer->SetFileName("out/output.jpg");

    caster->SetInput(resampler->GetOutput());
    writer->SetInput(caster->GetOutput());
    writer->Update();

    using DifferenceFilterType = itk::SubtractImageFilter<FixedImageType, FixedImageType, FixedImageType>;

    DifferenceFilterType::Pointer difference = DifferenceFilterType::New();

    difference->SetInput1(fixedImageReader->GetOutput());
    difference->SetInput2(resampler->GetOutput());

    WriterType::Pointer writer2 = WriterType::New();
    using RescalerType = itk::RescaleIntensityImageFilter<FixedImageType, OutputImageType>;

    RescalerType::Pointer intensityRescaler = RescalerType::New();

    intensityRescaler->SetInput(difference->GetOutput());
    intensityRescaler->SetOutputMinimum(0);
    intensityRescaler->SetOutputMaximum(255);

    writer2->SetInput(intensityRescaler->GetOutput());
    resampler->SetDefaultPixelValue(1);

    writer2->SetFileName("out/output2.jpg");
    writer2->Update();

    using IdentityTransformType = itk::IdentityTransform<double, Dimension>;
    IdentityTransformType::Pointer identity = IdentityTransformType::New();

    resampler->SetTransform(identity);
    writer2->SetFileName("out/output3.jpg");
    writer2->Update();

    plot = cv::plot::Plot2d::create(xData, yData);
    plot->setPlotSize(1000, 1000);
    plot->setMaxX(1500);
    plot->setMinX(000);
    plot->setMaxY(4000);
    plot->setMinY(2000);
    plot->render(display);
    //cv::imshow("Plot", display);

    using MatrixType = itk::Matrix<double, 2, 2>;
    MatrixType matrix;
    matrix = transform->GetMatrix();
    std::cout << "------------" << std::endl;
    std::cout << "---Matrix---" << std::endl;
    std::cout << matrix << std::endl;
    return transform;
}

PointType transform_point(PointType point, TransformType::Pointer transform)
{
    TransformType::InverseTransformBasePointer inv_transform = transform->GetInverseTransform();
    PointType out_point = inv_transform->TransformPoint(point);
    return out_point;
}