/* recalage.hpp */
#ifndef RECALAGE_HPP
#define RECALAGE_HPP
#include "itkCommand.h"
#include "itkImage.h"
#include "itkImageRegistrationMethodv4.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"
#include <itkLBFGSOptimizerv4.h>
#include "opencv2/core/core.hpp"

constexpr unsigned int Dimension = 2;
using TransformType = itk::AffineTransform<double, Dimension>;
using PointType = itk::Point<double, Dimension>;

class CommandIterationUpdate : public itk::Command
{
public:
    using Self = CommandIterationUpdate;
    using Superclass = itk::Command;
    using Pointer = itk::SmartPointer<Self>;
    itkNewMacro(Self);

protected:
    CommandIterationUpdate() = default;

public:
    using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;

    using OptimizerPointer = const OptimizerType *;

    void
    Execute(itk::Object *caller, const itk::EventObject &event) override;

    void
    Execute(const itk::Object *object, const itk::EventObject &event) override;
};

TransformType::Pointer registrate_image(std::string filename1, std::string filename2);

PointType transform_point(PointType point, TransformType::Pointer transform);

#endif