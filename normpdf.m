function p = normpdf(x,mu,sigma)
p = 1./sqrt(2.*pi.*sigma.^2).*exp(-((x-mu).^2)./(2.*sigma.^2));
end